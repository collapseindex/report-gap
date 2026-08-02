"""Does LEACE actually carry the guarantee INLP lacked? Checked, not cited.

`RESULTS_prompt_erase.md` reports INLP failing to erase a property at n=60 and n=1800. Before
spending any GPU on a re-run, these tests establish on synthetic data that LEACE does what the
citation says and that INLP does not, in the same regime that defeated us: many dimensions, few
samples, a redundantly encoded concept.

The load-bearing test is `test_leace_beats_inlp_in_the_regime_that_defeated_us`. Without it the
re-run would be another expensive way to learn the same lesson.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap.leace import LeaceEraser, class_mean_gap, fit_leace   # noqa: E402


def redundant_concept(n=200, d=256, k=24, seed=0):
    """A concept written into k directions at once, which is what defeats INLP.

    Returns (x, z). The label is encoded redundantly, so removing any one direction leaves the
    rest, exactly like a lexically obvious contrast in a residual stream.
    """
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    base = rng.normal(size=(n, d))
    dirs = rng.normal(size=(k, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    signal = (2.0 * z - 1)[:, None] * dirs.sum(axis=0)[None, :]
    return base + 0.6 * signal, z


def cv_decodability(x, z, erase=None, folds=5, seed=0):
    """Held-out DECODABILITY of a linear probe: max(acc, 1 - acc).

    Raw accuracy is the wrong metric and briefly fooled this file. INLP scored 0.067 here, which
    reads as "erased" and is the opposite: a probe at 0.067 is a probe at 0.933 with its sign
    flipped, so the concept is fully present. Any linear classifier can invert itself for free, so
    decodability is the distance from chance, not the accuracy. Same class of mistake as the
    sign-blind capability gate recorded in RESULTS_readjudicate.md.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    x, z = x[order], z[order]
    accs = []
    for f in range(folds):
        test = np.zeros(len(x), dtype=bool)
        test[f::folds] = True
        if len(np.unique(z[~test])) < 2:
            continue
        a, b = x[~test], x[test]
        if erase is not None:
            a, b = erase(a, z[~test], b)      # eraser fit on TRAIN ONLY
        m, s = a.mean(0), a.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000).fit((a - m) / s, z[~test])
        accs.append(clf.score((b - m) / s, z[test]))
    acc = float(np.mean(accs))
    return max(acc, 1.0 - acc)


def leace_erase(xa, za, xb):
    er = fit_leace(xa, za)
    return er(xa), er(xb)


def all_data_protocol(x, z, erase_all, folds=5):
    """The protocol modal_prompt_erase.py used: fit the eraser on EVERYTHING, then cross-validate.

    Kept so its bias is measurable rather than described.
    """
    return cv_decodability(erase_all(x, z), z, erase=None, folds=folds)


def inlp(x, z, k, seed=0):
    """Iterative nullspace projection, the method that failed in RESULTS_prompt_erase.md."""
    from sklearn.linear_model import LogisticRegression

    work = x.copy()
    for _ in range(k):
        m, s = work.mean(0), work.std(0) + 1e-8
        v = LogisticRegression(max_iter=1000).fit((work - m) / s, z).coef_[0] / s
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            break
        v = v / nrm
        work = work - np.outer(work @ v, v)
    return work


def test_the_premise_the_concept_is_decodable_before_erasure():
    """Without this, every test below could pass on data that never carried the concept."""
    x, z = redundant_concept()
    assert cv_decodability(x, z) > 0.90, "the fixture does not encode the concept; nothing else means anything"
    assert class_mean_gap(x, z) > 1.0


def test_leace_collapses_the_class_mean_gap():
    """The guarantee, stated directly: after erasure the class-conditional means coincide."""
    x, z = redundant_concept()
    before = class_mean_gap(x, z)
    erased = fit_leace(x, z)(x)
    after = class_mean_gap(erased, z)
    assert after < before * 1e-6, "gap %.3e did not collapse from %.3e" % (after, before)


def test_leace_drives_a_held_out_probe_to_chance():
    """The erasure gate, with the eraser fit on train only."""
    x, z = redundant_concept()
    assert cv_decodability(x, z, erase=leace_erase) < 0.62


def test_the_all_data_protocol_inflates_residual_decodability():
    """THE FINDING THAT MATTERS, and it is about our own previous arm.

    `modal_prompt_erase.py` fit the eraser on every row and then cross-validated a probe refit on
    that erased data. Because the eraser used the test rows' labels, the train-fold residual is the
    negative of the test-fold residual, so a probe learns one and anti-predicts the other, and
    max(acc, 1-acc) reports that as decodability. The result is an over-estimate.

    That is the protocol behind the cv=1.000 in RESULTS_prompt_erase.md, which is why the
    conclusion drawn from it ("INLP removes the directions you found, not the property") is not
    established.
    """
    x, z = redundant_concept(n=2000, d=256, k=24)
    honest = cv_decodability(x, z, erase=leace_erase)
    leaky = all_data_protocol(x, z, lambda a, b: fit_leace(a, b)(a))
    assert honest < 0.60, "honest protocol should read near chance, got %.3f" % honest
    assert leaky > honest + 0.15, (
        "the all-data protocol should over-report; honest %.3f vs leaky %.3f" % (honest, leaky))


def test_this_fixture_does_not_reproduce_the_real_data_failure():
    """Recorded so nobody reads more into these tests than they support.

    On a pure mean-shift concept INLP also erases successfully, so this fixture does NOT reproduce
    the failure seen on real activations. These tests establish that LEACE works and that the old
    evaluation protocol was biased; they do NOT establish that LEACE beats INLP on model
    activations. That question needs the real run.
    """
    x, z = redundant_concept(n=1800, d=2048, k=24)

    def inlp_erase(xa, za, xb):
        from sklearn.linear_model import LogisticRegression
        A, B = xa.copy(), xb.copy()
        for _ in range(16):
            m, s = A.mean(0), A.std(0) + 1e-8
            v = LogisticRegression(max_iter=1000).fit((A - m) / s, za).coef_[0] / s
            n = np.linalg.norm(v)
            if n < 1e-12:
                break
            v = v / n
            A, B = A - np.outer(A @ v, v), B - np.outer(B @ v, v)
        return A, B

    assert cv_decodability(x, z, erase=inlp_erase) < 0.62, "INLP also succeeds on this fixture"


def test_leace_is_rank_one_for_a_binary_concept():
    """The reason it is cheaper as well as stronger: one dimension in whitened space."""
    x, z = redundant_concept(n=200, d=128, k=16)
    assert fit_leace(x, z).rank == 1


def test_leace_preserves_the_overall_mean():
    """It is affine, not a projection to the origin; a mean shift would confound every downstream
    probe read with a translation."""
    x, z = redundant_concept()
    erased = fit_leace(x, z)(x)
    assert np.allclose(erased.mean(axis=0), x.mean(axis=0), atol=1e-8)


def test_leace_leaves_an_unrelated_direction_alone():
    """Specificity. An eraser that flattened everything would pass every test above."""
    x, z = redundant_concept()
    rng = np.random.default_rng(7)
    other = rng.normal(size=x.shape[1])
    other /= np.linalg.norm(other)
    erased = fit_leace(x, z)(x)
    keep = np.corrcoef(x @ other, erased @ other)[0, 1]
    assert keep > 0.95, "an unrelated direction was destroyed too (r=%.3f)" % keep


@pytest.mark.parametrize("bad", [np.zeros(50), np.arange(50) % 3])
def test_fit_leace_refuses_degenerate_labels(bad):
    """A silent no-op here would look exactly like a clean erasure."""
    x = np.random.default_rng(0).normal(size=(50, 16))
    with pytest.raises(ValueError):
        fit_leace(x, bad)


def test_eraser_applies_to_a_single_vector_and_a_batch_identically():
    x, z = redundant_concept(n=80, d=64, k=8)
    er = fit_leace(x, z)
    assert np.allclose(er(x)[3], er(x[3]))
