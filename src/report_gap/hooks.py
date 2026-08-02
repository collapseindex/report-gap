"""Residual-stream injection.

The whole experiment rests on this file doing exactly what it says, so the two things that can
silently ruin it are handled explicitly rather than assumed:

  1. A hook attached to the wrong module does nothing and returns a clean null. `assert_active`
     exists so that no run starts without proving the intervention changes the stream.
  2. A hook that is never removed contaminates every later condition. `inject` is a context manager
     and removes itself on the way out, including on exception.

Injection adds `alpha * scale * d` to the residual stream at the output of layer `L_inject`, at
every position processed while the hook is active. `scale` is the item's own mean residual norm at
that layer measured under no injection, so `alpha` means the same thing across items and models.

Because the prompt is byte-identical across conditions for a given item, `scale` is a property of
the item alone: measure it once, reuse it for every alpha and every direction.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch


def layer_module(model: torch.nn.Module, index: int) -> torch.nn.Module:
    """Return the decoder layer at `index`.

    Args:
        model: A causal language model, or any module exposing a `.model.layers` or `.layers` list.
        index: Layer index. Negative indices count from the end.

    Returns:
        The layer module itself, which is what a forward hook attaches to.

    Raises:
        AttributeError: If no layer list can be found, rather than guessing and hooking nothing.
    """
    for path in ("model.layers", "layers", "model.decoder.layers", "transformer.h"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj[index]
    raise AttributeError(
        "no layer list found on %s; refusing to attach a hook to nothing" % type(model).__name__
    )


def n_layers(model: torch.nn.Module) -> int:
    """Return the number of decoder layers."""
    for path in ("model.layers", "layers", "model.decoder.layers", "transformer.h"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return len(obj)
    raise AttributeError("no layer list found on %s" % type(model).__name__)


def _split(output):
    """Return (hidden_states, rebuild) for a layer output that may be a tensor or a tuple."""
    if isinstance(output, tuple):
        return output[0], lambda h: (h,) + output[1:]
    return output, lambda h: h


@contextlib.contextmanager
def inject(
    model: torch.nn.Module,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    scale: float | torch.Tensor,
) -> Iterator[dict]:
    """Add `alpha * scale * direction` to the residual stream at `layer`, for the duration.

    Args:
        model: The model to hook.
        layer: Index of the decoder layer whose output is modified.
        direction: Unit-norm direction in residual space, shape (hidden_size,).
        alpha: Injection strength from the frozen grid. alpha=0 is an exact no-op.
        scale: The item's mean residual norm at this layer, measured under no injection. A float
            for a single item, or a 1-D tensor of per-row norms of shape (batch,) when several
            items are run together. Per-row scaling is what makes batching safe: `alpha` has to
            mean the same thing for every item in the batch, and items differ in residual norm.

    Yields:
        A dict with a `calls` counter, so a caller can assert the hook actually fired. A hook that
        never fires is the failure mode this whole module is defensive about.

    Raises:
        ValueError: If `scale` is a tensor of the wrong rank, which would broadcast into a
            silently wrong perturbation rather than an error.
    """
    state = {"calls": 0}
    if isinstance(scale, torch.Tensor):
        if scale.dim() != 1:
            raise ValueError("per-row scale must be 1-D of shape (batch,), got shape %s"
                             % (tuple(scale.shape),))
        # (batch, 1, hidden) so it broadcasts against (batch, positions, hidden)
        delta = alpha * scale.detach().reshape(-1, 1, 1) * direction.detach().reshape(1, 1, -1)
    else:
        delta = (alpha * scale) * direction.detach()

    def hook(_module, _args, output):
        state["calls"] += 1
        if alpha == 0.0:
            return output
        hidden, rebuild = _split(output)
        d = delta.to(hidden.device, hidden.dtype)
        if d.dim() == 3 and d.shape[0] != hidden.shape[0]:
            raise RuntimeError("per-row scale has %d rows but the batch has %d"
                               % (d.shape[0], hidden.shape[0]))
        return rebuild(hidden + d)

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@contextlib.contextmanager
def project_out(
    model: torch.nn.Module,
    layer: int,
    direction: torch.Tensor,
) -> Iterator[dict]:
    """Remove the component along `direction` from the residual stream at `layer`, for the duration.

    This is a projection, not the subtraction of a fixed vector: it removes whatever component along
    `direction` is present at that layer, including any the model itself produced. That distinction
    is the whole point of the erase arm. Subtracting the vector we injected would only undo our own
    edit; projecting removes the direction entirely, so anything the probe still reads afterwards is
    in a subspace orthogonal to it.

    One-shot at a single layer. A persistent erase applied at every subsequent layer would prevent
    the model from ever re-forming a component along `direction`, which confounds "the state was
    transformed" with "the state was continuously suppressed".

    Args:
        model: The model to hook.
        layer: Index of the decoder layer whose output is projected.
        direction: The direction to remove, any norm; it is unit-normalized internally.

    Yields:
        A dict with a `calls` counter, so a caller can assert the hook fired.

    Raises:
        ValueError: If `direction` has zero norm, where the projection is undefined and silently
            doing nothing would look like a clean erase.
    """
    state = {"calls": 0}
    norm = float(direction.norm())
    if norm < 1e-9:
        raise ValueError("cannot project out a zero-norm direction")
    unit = (direction.detach() / norm).reshape(-1)

    def hook(_module, _args, output):
        state["calls"] += 1
        hidden, rebuild = _split(output)
        u = unit.to(hidden.device, hidden.dtype)
        # h - (h . u) u, broadcasting over batch and position
        coeff = (hidden * u).sum(dim=-1, keepdim=True)
        return rebuild(hidden - coeff * u)

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@torch.no_grad()
def residual_norm(model: torch.nn.Module, inputs: dict, layer: int) -> float:
    """Mean L2 norm of the residual stream at `layer`, over positions, under no injection.

    Args:
        model: The model.
        inputs: Tokenized inputs, already on the right device.
        layer: Decoder layer index.

    Returns:
        Mean per-position residual norm as a float.
    """
    out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output, so layer i's output is hidden_states[i + 1]
    hidden = out.hidden_states[layer + 1]
    return float(hidden.norm(dim=-1).mean())


@torch.no_grad()
def assert_active(
    model: torch.nn.Module,
    inputs: dict,
    layer: int,
    direction: torch.Tensor,
    scale: float,
    alpha: float = 1.0,
    tol: float = 1e-4,
) -> dict:
    """Prove the intervention is real before trusting any result from it.

    Checks both halves of the claim, because each has a distinct failure mode:
      - alpha=0 must reproduce the unhooked logits exactly (otherwise the hook itself perturbs).
      - alpha>0 must change them (otherwise the hook is attached to nothing).

    Args:
        model: The model.
        inputs: Tokenized inputs on the right device.
        layer: Decoder layer index to inject at.
        direction: Unit-norm direction.
        scale: Item residual norm.
        alpha: Strength to use for the "does something" half.
        tol: Max tolerated logit drift for the no-op half.

    Returns:
        A dict of the measured drifts, for logging next to the run.

    Raises:
        RuntimeError: If the hook never fired, silently did nothing, or perturbed at alpha=0.
    """
    base = model(**inputs).logits

    with inject(model, layer, direction, 0.0, scale) as state:
        zero = model(**inputs).logits
    if state["calls"] == 0:
        raise RuntimeError("hook never fired at layer %d: it is attached to nothing" % layer)
    noop_drift = float((zero - base).abs().max())
    if noop_drift > tol:
        raise RuntimeError(
            "alpha=0 changed the logits by %.3g (tol %.3g): the hook is not a no-op when it must be"
            % (noop_drift, tol)
        )

    with inject(model, layer, direction, alpha, scale):
        moved = model(**inputs).logits
    live_drift = float((moved - base).abs().max())
    if live_drift <= tol:
        raise RuntimeError(
            "alpha=%.3g changed the logits by only %.3g: the injection is a no-op when it must not "
            "be, so any null from this configuration is an artifact" % (alpha, live_drift)
        )

    return {"noop_drift": noop_drift, "live_drift": live_drift, "hook_calls": state["calls"]}


@contextlib.contextmanager
def project_out_subspace(
    model: torch.nn.Module,
    layer: int,
    basis: torch.Tensor,
) -> Iterator[dict]:
    """Remove the component in the span of `basis` from the residual stream at `layer`.

    The k-dimensional generalization of `project_out`. `RESULTS_erase.md` erases ONE fitted
    direction, which the paper itself calls the weakest part of that result: removing one vector is
    not removing a concept, and LEACE (Belrose et al. 2023) and amnesic probing (Elazar et al.
    2021) both erase a subspace or
    a property rather than a direction. This is the iterative-nullspace form of that, so the erase
    arm can be re-asked at k = 1, 2, 4, 8 instead of k = 1 only.

    The basis is orthonormalized here rather than assumed orthonormal, because the directions come
    from an iterative fit and near-collinear rows would silently under-erase.

    Args:
        model: The model to hook.
        layer: Index of the decoder layer whose output is projected.
        basis: Shape (k, hidden). Any norm; orthonormalized internally. k=0 is an exact no-op.

    Yields:
        A dict with a `calls` counter and the rank actually removed, so a caller can assert the
        hook fired and that it erased as many dimensions as it was asked to.

    Raises:
        ValueError: If `basis` is not 2-D, or if orthonormalization collapses it to a lower rank
            than requested, where a silent under-erase would look like a clean one.
    """
    if basis.ndim != 2:
        raise ValueError("basis must be (k, hidden), got shape %s" % (tuple(basis.shape),))
    k_req = basis.shape[0]
    state = {"calls": 0, "rank": 0}
    if k_req == 0:
        yield state
        return

    q, r = torch.linalg.qr(basis.T.to(torch.float32), mode="reduced")   # (hidden, k), (k, k)
    # Rank must be read off R, not Q. Q's columns are orthonormal by construction even when the
    # input is rank-deficient, so testing Q would always report full rank and a collinear basis
    # would under-erase silently, which is exactly the failure that looks like survival.
    diag = torch.diagonal(r).abs()
    tol = float(diag.max()) * 1e-6 if diag.numel() else 0.0
    rank = int((diag > tol).sum().item())
    if rank < k_req:
        raise ValueError(
            "basis of %d rows has rank %d; erasing it would remove fewer dimensions than "
            "requested and look like a clean erase" % (k_req, rank))
    state["rank"] = rank

    def hook(_module, _args, output):
        state["calls"] += 1
        hidden, rebuild = _split(output)
        b = q.to(hidden.device, hidden.dtype)
        coeff = hidden @ b                      # (..., k)
        return rebuild(hidden - coeff @ b.T)

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


@contextlib.contextmanager
def apply_affine(
    model: torch.nn.Module,
    layer: int,
    proj: torch.Tensor,
    bias: torch.Tensor,
) -> Iterator[dict]:
    """Apply `(h - bias) @ proj.T + bias` to the residual stream at `layer`, for the duration.

    `project_out_subspace` is linear and cannot express a LEACE eraser, which is AFFINE: it
    preserves the overall mean instead of translating everything toward the origin. Using a linear
    hook for an affine eraser would add a constant shift to every downstream read, confounding the
    probe measurement with a translation.

    One-shot at a single layer, like the other erase hooks, so the model can re-form the component
    downstream. A persistent erase would confound "re-encoded" with "continuously suppressed".

    Args:
        model: The model to hook.
        layer: Index of the decoder layer whose output is transformed.
        proj: (hidden, hidden) linear part.
        bias: (hidden,) the mean added back.

    Yields:
        A dict with a `calls` counter, so a caller can assert the hook fired.

    Raises:
        ValueError: On a shape mismatch, where broadcasting would silently do something else.
    """
    if proj.ndim != 2 or proj.shape[0] != proj.shape[1]:
        raise ValueError("proj must be square (hidden, hidden), got %s" % (tuple(proj.shape),))
    if bias.ndim != 1 or bias.shape[0] != proj.shape[0]:
        raise ValueError("bias must be (hidden,) matching proj, got %s" % (tuple(bias.shape),))

    state = {"calls": 0}

    def hook(_module, _args, output):
        state["calls"] += 1
        hidden, rebuild = _split(output)
        p = proj.to(hidden.device, hidden.dtype)
        b = bias.to(hidden.device, hidden.dtype)
        return rebuild((hidden - b) @ p.T + b)

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()
