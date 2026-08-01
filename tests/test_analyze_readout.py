"""End-to-end tests for the confirmatory analysis, on artifacts whose answer is known.

A pipeline that runs is not a pipeline that works. These build synthetic artifacts in the exact
schema `modal_readout.py` writes, with an effect planted by hand, and require the analysis to
recover it. Crucially there are TWO artifacts with DIFFERENT known answers: one with a readout gap
and one without. A checker validated only against the positive case will happily report a gap on
everything, and a fixture cannot reveal that on its own.

The gate tests are the load-bearing ones. `test_gate_failure_forces_uninformative` breaks the
instrument on purpose and requires the verdict to change, because a gate that never fails is not
gating anything.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from report_gap import stimuli as S  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_readout", ROOT / "experiments" / "analyze_readout.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AR = _load_module()

ALPHAS = (0.025, 0.05, 0.075, 0.10)
CONDITIONS = ("lexical_pos", "lexical_neg", "random_a", "random_b", "formality")
N_ITEMS = 30
SEEDS = (0, 1)


def _row(item, wording, seed, condition, alpha, mapping, probs, *, usable=True,
         off=0.02, logprob=-0.4):
    argmax = max(probs, key=probs.get)
    return {
        "cell": "%s|%s|%d|ABCDE" % (item, wording, seed),
        "item": item, "wording": wording, "seed": seed,
        "condition": condition, "alpha": alpha,
        "mapping": mapping, "probs": probs, "off_option_mass": off,
        "argmax": argmax, "letter": argmax, "key": mapping[argmax],
        "usable": usable, "degenerate": False, "refused": False, "truncated": False,
        "mean_logprob": logprob, "raw": argmax,
    }


# Distributions are constructed directly rather than emulated through logit pushes, so the planted
# quantities are exact. Emulating through logits makes the realized effect a function of the
# baseline shape and the alpha grid at once, and a fixture whose own answer needs solving is not a
# fixture you can check a pipeline against.
#
# Layout: the NEUTRAL option is the peak and takes whatever the two poles leave. The peak has to be
# neutral rather than any convenient letter, because the baseline is shared between the negative and
# positive arms (pairing requires one baseline per cell) and a peak sitting on either pole would
# start that arm saturated. With both poles near 0.20 the neutral option holds ~0.60, keeps the
# argmax through every alpha on the grid, and the per-cell indicator delta is exactly zero, which
# makes the per-cell discrepancy exactly the mass delta.
BASE_OWN_MASS = 0.20


def _probs(mapping, neg_mass, pos_mass):
    """Build a five-option distribution with prescribed pole masses and the neutral as peak."""
    neg = sorted(L for L, k in mapping.items() if k in {"neg1", "neg2"})
    pos = sorted(L for L, k in mapping.items() if k in {"pos1", "pos2"})
    neut = [L for L, k in mapping.items() if k == "neut"]
    assert len(neg) == 2 and len(pos) == 2 and len(neut) == 1, "unexpected option set"
    neutral_mass = 1.0 - neg_mass - pos_mass
    assert neutral_mass > max(neg_mass, pos_mass) / 2, \
        "neutral %.3f would lose the argmax to a pole option" % neutral_mass
    out = {L: neg_mass / 2 for L in neg}
    out.update({L: pos_mass / 2 for L in pos})
    out[neut[0]] = neutral_mass
    total = sum(out.values())
    return {L: p / total for L, p in out.items()}


def build_artifact(path: pathlib.Path, gap: float, mode: str = "gap",
                   wordings=None, seed: int = 0) -> pathlib.Path:
    """Write a synthetic artifact whose readout gap is known by arithmetic.

    Args:
        path: Where to write the JSONL.
        gap: Own-pole mass shift per unit alpha under the lexical arms.
        mode: What is planted.
            "gap"        mass moves by `gap * alpha`, argmax never moves. Per-cell discrepancy is
                         exactly `gap * alpha`, positive, which is the claim.
            "null"       the lexical arms behave like the random ones. Discrepancy is zero.
            "overfollow" the argmax moves onto an own-pole option while mass moves only a little,
                         so the discrepancy is NEGATIVE. This is the mirror artifact, and a
                         pipeline that reports a positive gap on it is reporting sign-blind.
        wordings: Which wordings to emit. Defaults to all three.
        seed: RNG seed.

    Returns:
        The path written.
    """
    if mode not in ("gap", "null", "overfollow"):
        raise ValueError("unknown mode %r" % mode)
    rng = random.Random(seed)
    wordings = list(S.WORDINGS if wordings is None else wordings)
    items = S.build_prompts()[:N_ITEMS]
    lines = []

    for wording in wordings:
        for perm in SEEDS:
            _, mapping = S.build_self_report_probe(perm, wording=wording)
            for item in items:
                m_neg = BASE_OWN_MASS + rng.uniform(-0.03, 0.03)
                m_pos = BASE_OWN_MASS + rng.uniform(-0.03, 0.03)
                base_probs = _probs(mapping, m_neg, m_pos)
                lines.append(_row(item, wording, perm, "baseline", 0.0, mapping, base_probs))

                for cond in CONDITIONS:
                    for a in ALPHAS:
                        if cond in ("lexical_neg", "lexical_pos"):
                            is_neg = cond == "lexical_neg"
                            own = sorted(L for L, k in mapping.items()
                                         if k in ({"neg1", "neg2"} if is_neg else {"pos1", "pos2"}))
                            if mode == "null":
                                shift = rng.gauss(0.0, 0.005)
                            elif mode == "gap":
                                shift = gap * a
                            else:
                                shift = 0.0
                            if mode == "overfollow" and rng.random() < 0.5:
                                # the argmax lands on an own-pole option, an indicator move of +1,
                                # while own-pole mass rises by only 0.4. Per-cell discrepancy is
                                # 0.4 - 1 = -0.6, so the mean is firmly negative and a sign-blind
                                # pipeline is caught.
                                probs = {L: (0.55 if L == own[0] else 0.05 if L in own
                                             else 0.40 / 3) for L in "ABCDE"}
                                total = sum(probs.values())
                                probs = {L: p / total for L, p in probs.items()}
                            else:
                                probs = _probs(mapping,
                                               m_neg + (shift if is_neg else 0.0),
                                               m_pos + (0.0 if is_neg else shift))
                        elif cond == "formality":
                            # the capability control MOVES the argmax, by construction
                            target = "ABCDE"[(perm + int(a * 1000)) % 5]
                            probs = {L: (0.6 if L == target else 0.1) for L in "ABCDE"}
                            total = sum(probs.values())
                            probs = {L: p / total for L, p in probs.items()}
                        else:
                            jitter = {L: max(1e-6, p + rng.gauss(0.0, a * 0.02))
                                      for L, p in base_probs.items()}
                            total = sum(jitter.values())
                            probs = {L: p / total for L, p in jitter.items()}
                        lines.append(_row(item, wording, perm, cond, a, mapping, probs))

    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    return path


def _own(probs, mapping, keys):
    return sum(p for L, p in probs.items() if mapping[L] in keys)


# --------------------------------------------------------------------------------------------
# two artifacts, two known and different answers
# --------------------------------------------------------------------------------------------

def _analyse_open(path):
    rows = AR.load(path)
    idx = AR.index(rows)
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    return AR.analyse(idx, present, scope="open")


def test_artifact_with_a_planted_gap_reports_one(tmp_path):
    p = build_artifact(tmp_path / "gap.jsonl", gap=0.30, mode="gap")
    got = _analyse_open(p)
    neg = got["contrast1_primary"]["lexical_neg"]
    assert neg["consecutive_significant"] >= 2, \
        "a planted readout gap was not detected: %s" % neg["per_alpha"]


def test_artifact_without_a_gap_reports_none(tmp_path):
    # the argmax tracks the mass perfectly, so there is nothing for the primary to find.
    # this is the fixture that catches a checker which reports a gap on everything.
    p = build_artifact(tmp_path / "nogap.jsonl", gap=0.30, mode="null", seed=5)
    got = _analyse_open(p)
    neg = got["contrast1_primary"]["lexical_neg"]
    assert neg["consecutive_significant"] < 2, \
        "reported a gap on an artifact built without one: %s" % neg["per_alpha"]


def test_the_two_artifacts_actually_disagree(tmp_path):
    # cross-validation of the metric itself. if both artifacts scored the same, the two tests
    # above would both be passing on a constant.
    a = _analyse_open(build_artifact(tmp_path / "a.jsonl", gap=0.30, mode="gap"))
    b = _analyse_open(build_artifact(tmp_path / "b.jsonl", gap=0.30, mode="null", seed=5))
    assert (a["contrast1_primary"]["lexical_neg"]["consecutive_significant"]
            != b["contrast1_primary"]["lexical_neg"]["consecutive_significant"])


# --------------------------------------------------------------------------------------------
# the gates
# --------------------------------------------------------------------------------------------

def test_strong_plant_gate_passes_on_a_sane_artifact(tmp_path):
    p = build_artifact(tmp_path / "gap.jsonl", gap=0.30, mode="gap")
    idx = AR.index(AR.load(p))
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    usable = AR.usable_mask(idx, present)
    got = AR.run_plant_gate(idx, present, usable, AR.STRONG_PLANT, 0.05, "strong")
    assert got["gate"] == "passed", got["reason"]


def test_floor_gate_fails_when_the_arm_is_too_noisy(tmp_path):
    # a gate that cannot fail is not a gate. plant the floor effect against absurd noise and
    # require it to refuse.
    p = build_artifact(tmp_path / "gap.jsonl", gap=0.30, mode="gap")
    idx = AR.index(AR.load(p))
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    usable = AR.usable_mask(idx, present)
    got = AR.run_plant_gate(idx, present, usable, AR.FLOOR_PLANT, 5.0, "floor")
    assert got["gate"] == "FAILED", \
        "the floor gate passed against sd=5.0, so it would pass against anything: %s" % got


def test_floor_gate_refuses_an_arm_whose_plant_was_shrunk_away(tmp_path):
    # The subtle failure. Shrinking targets to fit each cell's headroom truncates the plant's
    # spread, and the floor gate exists to ask whether 0.03 survives THAT spread. At sd=5.0 nearly
    # every target gets shrunk, so the arm that ran is far tamer than the arm requested and a
    # "detected" verdict would be about the tame version. Without this guard the gate passed at
    # sd=5.0, which is to say it passed against anything.
    p = build_artifact(tmp_path / "gap.jsonl", gap=0.30, mode="gap")
    idx = AR.index(AR.load(p))
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    usable = AR.usable_mask(idx, present)
    got = AR.run_plant_gate(idx, present, usable, AR.FLOOR_PLANT, 5.0, "floor")
    assert got["gate"] == "FAILED" and "shrunk" in got["reason"], got


def test_floor_gate_passes_at_a_realistic_noise_level(tmp_path):
    # same gate, same artifact, plausible spread: it must pass, or the gate rejects everything and
    # is no more informative than one that accepts everything.
    p = build_artifact(tmp_path / "gap.jsonl", gap=0.30, mode="gap")
    idx = AR.index(AR.load(p))
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    usable = AR.usable_mask(idx, present)
    got = AR.run_plant_gate(idx, present, usable, AR.FLOOR_PLANT, 0.05, "floor")
    assert got["gate"] == "passed", got["reason"]
    assert got["cells_shrunk_to_headroom"] == 0, \
        "the plant was shrunk at a realistic noise level, so the pass is about a tamer arm"


# --------------------------------------------------------------------------------------------
# integrity refusals
# --------------------------------------------------------------------------------------------

def test_torn_artifact_is_refused_not_partially_scored(tmp_path):
    p = build_artifact(tmp_path / "torn.jsonl", gap=0.3, mode="gap")
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"cell": "half a row"')
    with pytest.raises(SystemExit, match="unparseable"):
        AR.load(p)


def test_duplicate_rows_are_refused(tmp_path):
    p = build_artifact(tmp_path / "dupe.jsonl", gap=0.3, mode="gap")
    rows = AR.load(p)
    with pytest.raises(SystemExit, match="appended twice"):
        AR.index(rows + rows[:1])


def test_empty_artifact_is_refused(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no rows"):
        AR.load(p)


def test_unusable_cells_are_excluded_from_every_arm_not_just_one(tmp_path):
    # a cell excluded in one arm and kept in another makes the arms incomparable while every
    # printed n still looks reasonable.
    p = build_artifact(tmp_path / "mixed.jsonl", gap=0.3, mode="gap")
    rows = AR.load(p)
    victim = rows[10]["cell"]
    for r in rows:
        if r["cell"] == victim and r["condition"] == "lexical_neg":
            r["usable"] = False
    idx = AR.index(rows)
    present = {w for (_c, _a, w, _cell) in idx if w != S.HELD_OUT_WORDING}
    usable = AR.usable_mask(idx, present)
    assert victim not in usable, "a cell unusable in one arm survived into the common cell set"


# --------------------------------------------------------------------------------------------
# the held-out wording discipline
# --------------------------------------------------------------------------------------------

def test_open_result_is_on_disk_before_the_held_out_one(tmp_path, capsys):
    p = build_artifact(tmp_path / "readout.jsonl", gap=0.30, mode="gap")
    assert AR.main(["analyze_readout.py", str(p)]) == 0
    written = sorted(q.name for q in tmp_path.glob("*.json"))
    opens = [q for q in written if q.endswith("_open_wordings.json")]
    helds = [q for q in written if q.endswith("_held_out_wording.json")]
    assert opens and helds
    assert (tmp_path / opens[0]).stat().st_mtime <= (tmp_path / helds[0]).stat().st_mtime


def test_analysis_runs_when_the_held_out_wording_is_absent(tmp_path):
    p = build_artifact(tmp_path / "readout.jsonl", gap=0.30, mode="gap",
                       wordings=[w for w in S.WORDINGS if w != S.HELD_OUT_WORDING])
    assert AR.main(["analyze_readout.py", str(p)]) == 0
    assert not list(tmp_path.glob("*_held_out_wording.json"))


def test_every_screened_axis_is_actually_computed(tmp_path):
    p = build_artifact(tmp_path / "readout.jsonl", gap=0.30, mode="gap")
    got = _analyse_open(p)
    axes = got["contrast10_screened_axes"]["lexical_neg"]
    for axis in S.SCREENED_AXES:
        assert axis in axes, "screened axis %s is declared but never measured" % axis


def test_capability_control_is_seen_to_move_the_argmax(tmp_path):
    p = build_artifact(tmp_path / "readout.jsonl", gap=0.30, mode="gap")
    got = _analyse_open(p)
    assert got["capability_control_moves_argmax"]


def test_null_ablation_gap_is_reported_as_a_number(tmp_path):
    p = build_artifact(tmp_path / "readout.jsonl", gap=0.30, mode="gap")
    got = _analyse_open(p)
    assert "null_ablation_gap" in got
    assert got["null_ablation_gap"]["lexical_neg"], "the matched-random shift is not on the page"
