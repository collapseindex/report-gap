"""Tests for residual injection and direction fitting.

These run against a hand-built fake model rather than a downloaded one, on purpose: the mechanics
that can silently ruin the experiment (a hook attached to nothing, a hook that never detaches, an
alpha=0 that is not a no-op) are properties of the hook code, not of any particular checkpoint, and
a test that needs six gigabytes of weights is a test that gets skipped.

`experiments/selftest.py` covers the same chain end to end on a real small model.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from torch import nn

from report_gap import directions as D
from report_gap import hooks as H

HIDDEN = 16


class _Layer(nn.Module):
    """A decoder layer that returns a bare tensor, as transformers v5 layers do."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return x + self.lin(x)


class _TupleLayer(_Layer):
    """A decoder layer that returns a tuple, as older layers do. Both must be handled."""

    def forward(self, x):
        return (x + self.lin(x),)


class _FakeModel(nn.Module):
    """Minimal stand-in exposing the `.model.layers` path the hook helpers navigate."""

    def __init__(self, n_layers: int = 4, tuple_output: bool = False, run_layers: bool = True):
        super().__init__()
        cls = _TupleLayer if tuple_output else _Layer
        self.model = types.SimpleNamespace(layers=nn.ModuleList([cls() for _ in range(n_layers)]))
        # registered so .parameters() sees them and device lookup works
        self.layers_holder = self.model.layers
        self.embed = nn.Embedding(32, HIDDEN)
        self.head = nn.Linear(HIDDEN, 32)
        self.run_layers = run_layers

    def forward(self, input_ids, output_hidden_states=False, **_):
        h = self.embed(input_ids)
        hidden_states = [h]
        if self.run_layers:
            for layer in self.model.layers:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
                hidden_states.append(h)
        return types.SimpleNamespace(
            logits=self.head(h),
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = _FakeModel().eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    direction = torch.tensor(D.random_direction(HIDDEN, seed=0))
    return model, inputs, direction


# --------------------------------------------------------------------------------------------
# locating the layer
# --------------------------------------------------------------------------------------------


def test_layer_module_finds_the_layer(setup):
    model, _, _ = setup
    assert H.layer_module(model, 0) is model.model.layers[0]
    assert H.layer_module(model, -1) is model.model.layers[-1]
    assert H.n_layers(model) == 4


def test_layer_module_raises_rather_than_hooking_nothing():
    with pytest.raises(AttributeError, match="refusing to attach"):
        H.layer_module(nn.Linear(2, 2), 0)


# --------------------------------------------------------------------------------------------
# the two halves of "the intervention is real"
# --------------------------------------------------------------------------------------------


def test_alpha_zero_is_an_exact_noop(setup):
    model, inputs, direction = setup
    with torch.no_grad():
        base = model(**inputs).logits
        with H.inject(model, 2, direction, 0.0, scale=1.0) as state:
            got = model(**inputs).logits
    assert state["calls"] == 1
    assert torch.equal(base, got)


def test_nonzero_alpha_actually_changes_the_stream(setup):
    model, inputs, direction = setup
    with torch.no_grad():
        base = model(**inputs).logits
        with H.inject(model, 2, direction, 1.0, scale=1.0):
            got = model(**inputs).logits
    assert (got - base).abs().max() > 1e-4


def test_injection_scales_with_alpha(setup):
    model, inputs, direction = setup
    drifts = []
    with torch.no_grad():
        base = model(**inputs).logits
        for alpha in (0.25, 0.5, 1.0, 2.0):
            with H.inject(model, 2, direction, alpha, scale=1.0):
                drifts.append(float((model(**inputs).logits - base).abs().max()))
    assert drifts == sorted(drifts), "drift is not monotone in alpha: %r" % drifts


def test_tuple_returning_layers_are_handled():
    torch.manual_seed(0)
    model = _FakeModel(tuple_output=True).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    direction = torch.tensor(D.random_direction(HIDDEN, seed=1))
    with torch.no_grad():
        base = model(**inputs).logits
        with H.inject(model, 1, direction, 1.0, scale=1.0):
            got = model(**inputs).logits
    assert (got - base).abs().max() > 1e-4


# --------------------------------------------------------------------------------------------
# the hook must not outlive its context, or every later condition is contaminated
# --------------------------------------------------------------------------------------------


def test_hook_is_removed_on_exit(setup):
    model, inputs, direction = setup
    with torch.no_grad():
        base = model(**inputs).logits
        with H.inject(model, 2, direction, 1.0, scale=1.0):
            pass
        after = model(**inputs).logits
    assert torch.equal(base, after)


def test_hook_is_removed_even_on_exception(setup):
    model, inputs, direction = setup
    with torch.no_grad():
        base = model(**inputs).logits
    with pytest.raises(ValueError):
        with H.inject(model, 2, direction, 1.0, scale=1.0):
            raise ValueError("boom")
    with torch.no_grad():
        after = model(**inputs).logits
    assert torch.equal(base, after)


# --------------------------------------------------------------------------------------------
# assert_active is the gate that stops a silent no-op becoming a clean null
# --------------------------------------------------------------------------------------------


def test_assert_active_passes_on_a_working_setup(setup):
    model, inputs, direction = setup
    report = H.assert_active(model, inputs, 2, direction, scale=1.0)
    assert report["noop_drift"] == 0.0
    assert report["live_drift"] > 1e-4
    assert report["hook_calls"] >= 1


def test_assert_active_catches_a_hook_that_never_fires():
    torch.manual_seed(0)
    model = _FakeModel(run_layers=False).eval()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    direction = torch.tensor(D.random_direction(HIDDEN, seed=0))
    with pytest.raises(RuntimeError, match="never fired"):
        H.assert_active(model, inputs, 1, direction, scale=1.0)


def test_assert_active_catches_an_injection_that_does_nothing(setup):
    model, inputs, _ = setup
    with pytest.raises(RuntimeError, match="no-op when it must not be"):
        H.assert_active(model, inputs, 2, torch.zeros(HIDDEN), scale=1.0)


def test_assert_active_catches_a_scale_of_zero(setup):
    model, inputs, direction = setup
    with pytest.raises(RuntimeError, match="no-op when it must not be"):
        H.assert_active(model, inputs, 2, direction, scale=0.0)


# --------------------------------------------------------------------------------------------
# residual norm
# --------------------------------------------------------------------------------------------


def test_residual_norm_is_positive_and_layer_specific(setup):
    model, inputs, _ = setup
    norms = [H.residual_norm(model, inputs, layer) for layer in range(4)]
    assert all(n > 0 for n in norms)
    assert len(set(norms)) > 1, "every layer reported the same norm, indexing is probably wrong"


# --------------------------------------------------------------------------------------------
# direction fitting
# --------------------------------------------------------------------------------------------


def _planted(n_per_group: int = 8, dim: int = 32, strength: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    planted = rng.standard_normal(dim)
    planted /= np.linalg.norm(planted)
    acts, labels, groups = [], [], []
    for g in range(4):
        for i in range(n_per_group):
            label = i % 2
            sign = 1.0 if label else -1.0
            acts.append(sign * strength * planted + rng.standard_normal(dim))
            labels.append(label)
            groups.append("g%d" % g)
    return np.array(acts, dtype=np.float32), np.array(labels), np.array(groups), planted


def test_fit_direction_recovers_a_planted_direction():
    acts, labels, groups, planted = _planted()
    fitted = D.fit_direction(acts, labels, groups, layer=3)
    assert fitted.cv_accuracy > 0.85, "probe cannot recover a strongly planted axis"
    assert D.cosine(fitted.vector, planted) > 0.7
    assert np.isclose(np.linalg.norm(fitted.vector), 1.0, atol=1e-5)


def test_diffmeans_also_recovers_it_and_the_two_differ():
    acts, labels, groups, planted = _planted()
    disc = D.fit_direction(acts, labels, groups, layer=3, method="discriminative")
    means = D.fit_direction(acts, labels, groups, layer=3, method="diffmeans")
    assert D.cosine(means.vector, planted) > 0.7
    assert D.cosine(disc.vector, means.vector) < 0.999, "the two methods collapsed to one vector"


def test_probe_is_at_chance_when_there_is_nothing_to_find():
    rng = np.random.default_rng(1)
    acts = rng.standard_normal((32, 32)).astype(np.float32)
    labels = np.array([i % 2 for i in range(32)])
    groups = np.array(["g%d" % (i // 8) for i in range(32)])
    fitted = D.fit_direction(acts, labels, groups, layer=3)
    assert fitted.cv_accuracy < 0.75, "probe found structure in noise: %.2f" % fitted.cv_accuracy


def test_fit_direction_needs_at_least_two_groups():
    acts, labels, _, _ = _planted()
    with pytest.raises(ValueError, match="two frame groups"):
        D.fit_direction(acts, labels, np.array(["only"] * len(labels)), layer=3)


def test_random_direction_is_unit_norm_and_seeded():
    a = D.random_direction(128, seed=7)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-6)
    assert np.array_equal(a, D.random_direction(128, seed=7))
    assert not np.array_equal(a, D.random_direction(128, seed=8))


def test_cosine_floor_is_small_but_not_zero():
    mean, worst = D.random_cosine_floor(2048, n=32, seed=0)
    assert 0.0 < mean < 0.05, "unexpected floor %.4f" % mean
    assert worst >= mean
