"""The k-dimensional erasure hook, and the prompt-induced contexts.

The load-bearing tests are the two that could let a broken erase look like a clean one: a
rank-deficient basis must raise rather than under-erase, and erasing a subspace must actually leave
the stream orthogonal to every row of it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from report_gap import hooks as H          # noqa: E402
from report_gap import stimuli as S        # noqa: E402


class Tiny(torch.nn.Module):
    """A two-layer stand-in whose layer output is a known tensor, so erasure is checkable."""

    def __init__(self, hidden=16):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(2)])
        self.hidden = hidden


def _run(model, layer, x):
    return model.model.layers[layer](x)


def test_erasing_a_subspace_leaves_the_stream_orthogonal_to_every_row():
    torch.manual_seed(0)
    hidden, k = 16, 4
    model = Tiny(hidden)
    basis = torch.randn(k, hidden)
    x = torch.randn(3, 5, hidden)

    with H.project_out_subspace(model, 0, basis) as state:
        out = _run(model, 0, x)
    assert state["calls"] == 1, "the hook never fired, so this test proves nothing"
    assert state["rank"] == k

    q, _ = torch.linalg.qr(basis.T.to(torch.float32), mode="reduced")
    residual = out.to(torch.float32) @ q
    assert residual.abs().max() < 1e-4, (
        "the stream still has a component in the erased subspace: max %.3g"
        % residual.abs().max())


def test_the_premise_holds_before_erasure():
    """Negative control: if x were already orthogonal to the basis, the test above is vacuous."""
    torch.manual_seed(0)
    hidden, k = 16, 4
    basis = torch.randn(k, hidden)
    x = torch.randn(3, 5, hidden)
    q, _ = torch.linalg.qr(basis.T.to(torch.float32), mode="reduced")
    assert (x.to(torch.float32) @ q).abs().max() > 0.1, \
        "the input was already nearly orthogonal to the basis"


def test_k_zero_is_an_exact_no_op():
    model = Tiny(8)
    x = torch.randn(2, 3, 8)
    with H.project_out_subspace(model, 0, torch.zeros(0, 8)) as state:
        out = _run(model, 0, x)
    assert state["calls"] == 0
    assert torch.equal(out, x)


def test_rank_deficient_basis_raises_rather_than_under_erasing():
    """A silently under-erasing hook produces survival that looks like a finding."""
    v = torch.randn(1, 12)
    basis = torch.cat([v, 2.0 * v, torch.randn(1, 12)], dim=0)   # row 2 is collinear with row 1
    model = Tiny(12)
    with pytest.raises(ValueError, match="rank"):
        with H.project_out_subspace(model, 0, basis):
            pass


def test_non_2d_basis_raises():
    model = Tiny(8)
    with pytest.raises(ValueError):
        with H.project_out_subspace(model, 0, torch.randn(8)):
            pass


def test_erasing_more_dimensions_removes_at_least_as_much():
    """Monotonicity. If k=8 left more signal than k=1, the hook would be wrong."""
    torch.manual_seed(1)
    hidden = 32
    model = Tiny(hidden)
    x = torch.randn(4, 6, hidden)
    norms = []
    for k in (1, 2, 4, 8):
        torch.manual_seed(7)
        basis = torch.randn(k, hidden)
        with H.project_out_subspace(model, 0, basis):
            out = _run(model, 0, x)
        norms.append(float(out.norm()))
    # each erase uses a different random basis, so this is a weak monotonicity check on the mean
    assert norms[-1] <= norms[0] + 1e-3, "erasing 8 dimensions left more norm than erasing 1"


# ---------------------------------------------------------------- prompt-induced contexts

def test_the_three_framings_differ_only_in_the_middle_clause():
    a = S.build_prompt_induced("aversive", 3)
    n = S.build_prompt_induced("neutral", 3)
    p = S.build_prompt_induced("pleasant", 3)
    for x, y, z in zip(a, n, p):
        head = "You are partway through reviewing"
        tail = "You have finished about half of it. Here is where you are."
        for s in (x, y, z):
            assert s.startswith(head) and s.endswith(tail)
        assert x != y != z and x != z


def test_the_framings_are_length_matched():
    """A valence effect that is really a length effect would be indistinguishable otherwise."""
    a = S.build_prompt_induced("aversive")
    p = S.build_prompt_induced("pleasant")
    for x, y in zip(a, p):
        assert abs(len(x) - len(y)) <= 8, \
            "aversive and pleasant differ by %d characters, enough to confound" % abs(len(x) - len(y))


def test_unknown_framing_raises():
    with pytest.raises(KeyError):
        S.build_prompt_induced("ecstatic")


def test_prompt_induced_covers_every_review_topic():
    assert len(S.build_prompt_induced("neutral")) == len(S.REVIEW_CONTEXTS)


def test_adding_these_stimuli_did_not_change_the_enumerate_hash():
    import json
    header = ROOT / "data" / "enum_instruct" / "header.json"
    if not header.exists():
        pytest.skip("enumerate artifact not present")
    recorded = json.loads(header.read_text(encoding="utf-8"))["stimuli_sha256"]
    assert S.frozen_hash("enumerate") == recorded


# ---------------------------------------------------------------- the scaled-up contexts

def test_large_build_is_matched_pairwise():
    """Element i of one framing must be the matched partner of element i of another."""
    a = S.build_prompt_induced_large("aversive")
    n = S.build_prompt_induced_large("neutral")
    p = S.build_prompt_induced_large("pleasant")
    assert len(a) == len(n) == len(p) == 900
    for x, y in zip(a, p):
        assert abs(len(x) - len(y)) <= 12, "pair differs by %d chars" % abs(len(x) - len(y))
        assert x.split(" for a colleague.")[0] == y.split(" for a colleague.")[0]
        assert x.endswith("Here is where you are.")


def test_large_build_has_no_duplicates():
    """A duplicated context inflates n without adding information, which is the failure this
    whole re-run exists to avoid."""
    for f in ("aversive", "neutral", "pleasant"):
        got = S.build_prompt_induced_large(f)
        assert len(set(got)) == len(got), "%s has %d duplicates" % (f, len(got) - len(set(got)))


def test_large_build_varies_the_clause_not_just_the_topic():
    """If every context used one valence phrase, a probe could memorize it and n would not help."""
    a = S.build_prompt_induced_large("aversive")
    clauses = {x.split(" for a colleague. ")[1].split(" You have finished")[0]
               .split(" You are nearly")[0] for x in a}
    assert len(clauses) >= len(S.PROMPT_CLAUSE_TRIPLES),         "only %d distinct valence clauses" % len(clauses)


def test_large_build_stages_are_shared_across_framings():
    """The stage phrases must be valence-neutral, i.e. identical in all three framings."""
    for f in ("aversive", "neutral", "pleasant"):
        got = " ".join(S.build_prompt_induced_large(f, 1))
        for stage in S.PROMPT_STAGES:
            assert stage in got, "%s is missing stage %r" % (f, stage)


def test_large_build_rejects_unknown_framing():
    with pytest.raises(KeyError):
        S.build_prompt_induced_large("euphoric")


def test_large_build_n_exceeds_the_old_one_by_an_order_of_magnitude():
    """The whole point of the re-run. n=60 made the erasure check unable to fail."""
    assert len(S.build_prompt_induced_large("aversive")) >= 10 * len(
        S.build_prompt_induced("aversive"))
