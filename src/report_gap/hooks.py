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
