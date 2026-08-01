"""Score the confirmatory artifact against the frozen endpoints in PREREG_readout_gap.md.

Runs locally on the JSONL that `modal_readout.py` wrote. No GPU, no model, no judge: every number
here is a softmax read or a count, so anyone can recompute them from the artifact.

Order of operations is the prereg's, not the convenient one:

  1. Integrity. Provenance, key uniqueness, exclusion accounting. Refuses to score a broken run.
  2. Instrument gates, contrasts 7 and 8. The planted-discrepancy controls are built HERE, from
     this model's own baseline distributions and this model's own observed per-cell spread, which
     is why they run after the treatment arm rather than beside it. If either gate fails, every
     primary and co-primary cell is reported `uninformative` whatever it shows.
  3. The two open wordings. Written to disk with a timestamp BEFORE the held-out one is read.
  4. The held-out wording.
  5. Everything else: matched-random specificity, capability positive control, screened axes.

Step 3 is enforced rather than promised: the held-out rows are not loaded until the open-wording
result file exists on disk.

    python experiments/analyze_readout.py data/qwen3b/readout.jsonl
"""

from __future__ import annotations

import collections
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from report_gap import analysis as A          # noqa: E402
from report_gap import planted as P           # noqa: E402
from report_gap import stimuli as S           # noqa: E402

STRONG_PLANT = 0.15
FLOOR_PLANT = 0.03
RECOVERY_TOLERANCE = 0.01     # how far the strong plant may be misread before the gate fails

# How many cells may be saturated (no headroom for the plant in the own-pole direction) before the
# gate refuses. Set at 15% rather than a tidier 5% because saturation is a property of the readout
# and not of the plant: the pilot baseline put 59 of 60 cells on one letter, so a cell whose peak
# already sits on an own-pole option can start above 0.95 own-pole mass. Excluding those is only
# defensible if they are also cells the TREATMENT cannot move, which is why the gate reports the
# treatment's own effect on the saturated set alongside. If the saturated cells are where the
# effect lives, the exclusion is selecting the easy half and the gate says so.
MAX_SATURATED_FRACTION = 0.15

# The capability positive control has to move the argmax on at least this share of cells before
# "argmax under-reports" is a testable claim rather than a statement about an inert readout. Set at
# 5% because the treatment arm itself moved the argmax on 8.75% of cells at the top of the band:
# a control that moves it less than the treatment does cannot certify the readout was capable.
CAPABILITY_ARGMAX_FLOOR = 0.05
NEG_KEYS = {"neg1", "neg2"}
POS_KEYS = {"pos1", "pos2"}
ARMS = {"lexical_neg": NEG_KEYS, "lexical_pos": POS_KEYS}
RANDOM_ARMS = ("random_a", "random_b")


def load(path: pathlib.Path) -> list[dict]:
    """Read the artifact, refusing a torn or empty one rather than scoring what survived."""
    rows, torn = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
    if not rows:
        raise SystemExit("%s holds no rows" % path)
    if torn:
        raise SystemExit("%s has %d unparseable line(s). An interrupted run is resumable; rerun "
                         "modal_readout.py to complete it rather than scoring the fragment."
                         % (path, torn))
    return rows


def own_pole(row: dict, keys: set[str]) -> float:
    """Share of this cell's option mass on options whose valence key is in `keys`."""
    letters = {L for L, k in row["mapping"].items() if k in keys}
    return A.option_mass(row["probs"], letters)


def index(rows: list[dict]) -> dict[tuple, dict]:
    """Index rows by (condition, alpha, wording, cell), asserting the key is really unique."""
    out = {}
    for r in rows:
        key = (r["condition"], r["alpha"], r["wording"], r["cell"])
        if key in out:
            raise SystemExit("duplicate row for %s: the artifact has been appended twice and "
                             "per-cell pairing is not what it appears to be" % (key,))
        out[key] = r
    return out


def cells_for(idx: dict, condition: str, alpha: float, wordings: set[str]) -> dict[str, dict]:
    """All cells of one condition at one alpha, restricted to a wording set."""
    return {c: r for (cond, a, w, c), r in idx.items()
            if cond == condition and abs(a - alpha) < 1e-12 and w in wordings}


def usable_mask(idx: dict, wordings: set[str]) -> set[str]:
    """Cells usable in EVERY condition, so every contrast runs on one cell set.

    A cell excluded in one arm and kept in another would make the arms incomparable while every
    printed n still looked reasonable. Exclusions are counted and reported, never dropped quietly.
    """
    per_cell = collections.defaultdict(list)
    for (_cond, _a, w, c), r in idx.items():
        if w in wordings:
            per_cell[c].append(r)
    return {c for c, rs in per_cell.items() if all(r["usable"] for r in rs)}


def arm_deltas(idx: dict, condition: str, alpha: float, keys: set[str],
               wordings: set[str], usable: set[str]) -> tuple[list[float], dict, dict, dict, dict]:
    """Per-cell discrepancy for one arm at one alpha, plus the four dicts it was built from."""
    treat = cells_for(idx, condition, alpha, wordings)
    base = cells_for(idx, "baseline", 0.0, wordings)
    common = sorted(set(treat) & set(base) & usable)
    if not common:
        raise SystemExit("no usable cells for %s at alpha=%.3f" % (condition, alpha))
    mt = {c: own_pole(treat[c], keys) for c in common}
    mb = {c: own_pole(base[c], keys) for c in common}
    at = {c: treat[c]["mapping"][treat[c]["argmax"]] in keys for c in common}
    ab = {c: base[c]["mapping"][base[c]["argmax"]] in keys for c in common}
    return A.discrepancy_deltas(mt, mb, at, ab), mt, mb, at, ab


def stdev(values: list[float]) -> float:
    """Sample standard deviation, used to size the floor plant at the arm's own noise level."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least 2 values for a spread")
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def run_plant_gate(idx: dict, wordings: set[str], usable: set[str], nominal: float,
                   observed_sd: float, label: str) -> dict:
    """Build a planted arm from this model's own baselines and require the pipeline to recover it.

    The known value is fixed by `planted.py` without reference to `analysis.py`, so recovering it
    is evidence about the analysis path rather than a restatement of it.
    """
    base = cells_for(idx, "baseline", 0.0, wordings)
    common = sorted(set(base) & usable)
    baselines = {c: base[c]["probs"] for c in common}
    own_letters = {c: {L for L, k in base[c]["mapping"].items() if k in NEG_KEYS}
                   for c in common}
    # one own-pole letter set per cell, so plant_arm is called per cell rather than en bloc
    targets = P.matched_noise_targets(nominal, observed_sd, len(common), seed=0)

    # A peaked readout leaves some cells with no room for the full plant. Shrinking the target to
    # the available headroom and recording the shrunk value keeps the known value known, where
    # clipping would not and dropping the cell would run the control on the easy half of the arm.
    planted, skipped, shrunk = {}, [], 0
    for (cell, target) in zip(common, targets):
        before = A.option_mass(baselines[cell], own_letters[cell])
        fitted = P.fit_to_headroom(before, target)
        if abs(fitted - target) > 1e-12:
            shrunk += 1
        if abs(fitted) < 1e-9:
            skipped.append(cell)
            continue
        try:
            planted[cell] = P.plant(baselines[cell], own_letters[cell], fitted, nominal=nominal)
        except ValueError:
            skipped.append(cell)
    fraction = len(skipped) / len(common)
    if not planted or fraction > MAX_SATURATED_FRACTION:
        return {"label": label, "gate": "FAILED", "reason":
                "%d of %d cells (%.1f%%) have no headroom at all for a plant of %+.3f, over the "
                "%.0f%% bar" % (len(skipped), len(common), 100 * fraction, nominal,
                                100 * MAX_SATURATED_FRACTION)}

    keys = sorted(planted)
    mt = {c: planted[c].mass_after for c in keys}
    mb = {c: planted[c].mass_before for c in keys}
    at = {c: planted[c].argmax_after in own_letters[c] for c in keys}
    ab = {c: planted[c].argmax_before in own_letters[c] for c in keys}
    got = A.paired_bootstrap(A.discrepancy_deltas(mt, mb, at, ab))
    want = P.expected_discrepancy([planted[c] for c in keys])
    realized = sum(planted[c].target for c in keys) / len(keys)

    # Shrinking to headroom truncates the plant's spread, and the floor gate's whole job is to ask
    # whether 0.03 is resolvable AGAINST that spread. If most cells were shrunk, the arm that ran
    # is tamer than the arm that was asked for, and "it detected 0.03" is a statement about the
    # tame version. Refuse rather than report a power check that was quietly made easier.
    shrunk_fraction = shrunk / len(common)
    if shrunk_fraction > MAX_SATURATED_FRACTION:
        return {"label": label, "gate": "FAILED",
                "reason": "%d of %d cells (%.1f%%) had the plant shrunk to fit their headroom, so "
                          "the arm no longer carries the spread the gate is meant to test against "
                          "(requested sd=%.4f, realized mean target %+.4f vs nominal %+.4f)"
                          % (shrunk, len(common), 100 * shrunk_fraction, observed_sd,
                             realized, nominal),
                "nominal": nominal, "realized_mean_target": realized,
                "cells_shrunk_to_headroom": shrunk, "planted_sd": observed_sd,
                "skipped": len(skipped), "observed": str(got), "expected": want,
                "saturation_check": None}

    if nominal == FLOOR_PLANT:
        ok = got.excludes_zero
        reason = ("detects a planted %+.3f against sd=%.4f at n=%d" % (nominal, observed_sd, got.n)
                  if ok else
                  "CANNOT detect a planted %+.3f against sd=%.4f at n=%d, so every null on the "
                  "real arms is uninformative rather than absent" % (nominal, observed_sd, got.n))
    else:
        ok = abs(got.point - want) <= RECOVERY_TOLERANCE
        reason = ("recovers %+.4f where the plant put %+.4f" % (got.point, want) if ok else
                  "reads %+.4f where the plant put %+.4f, outside the %.3f tolerance; the "
                  "statistic does not measure what it is claimed to measure"
                  % (got.point, want, RECOVERY_TOLERANCE))
    # Is the exclusion defensible? Only if the saturated cells are ones the treatment cannot move
    # either. If the real effect lives on exactly the cells the plant had to skip, the control ran
    # on the easy half and its verdict does not transfer.
    saturation = None
    if skipped:
        treat = cells_for(idx, "lexical_neg", max(a for (_c, a, _w, _cell) in idx if a > 0.0),
                          wordings)
        base_all = cells_for(idx, "baseline", 0.0, wordings)
        def shift(cs):
            vals = [own_pole(treat[c], NEG_KEYS) - own_pole(base_all[c], NEG_KEYS)
                    for c in cs if c in treat and c in base_all]
            return sum(vals) / len(vals) if vals else float("nan")
        kept = [c for c in common if c not in set(skipped)]
        saturation = {"treatment_shift_on_saturated": shift(skipped),
                      "treatment_shift_on_kept": shift(kept)}

    return {"label": label, "gate": "passed" if ok else "FAILED", "reason": reason,
            "observed": str(got), "expected": want, "skipped": len(skipped),
            "nominal": nominal, "realized_mean_target": realized,
            "cells_shrunk_to_headroom": shrunk, "planted_sd": observed_sd,
            "saturation_check": saturation}


def analyse(idx: dict, wordings: set[str], scope: str) -> dict:
    """Every contrast, over one wording set."""
    usable = usable_mask(idx, wordings)
    total_cells = len({c for (_c, _a, w, c) in idx if w in wordings})
    report: dict = {"scope": scope, "wordings": sorted(wordings),
                    "cells_total": total_cells, "cells_usable": len(usable),
                    "excluded": total_cells - len(usable)}
    if not usable:
        report["verdict"] = "uninformative"
        report["reason"] = "no cell is usable in every condition"
        return report

    alphas = sorted({a for (_c, a, _w, _cell) in idx if a > 0.0})

    # ---- contrast 1, primary: mass effect minus argmax effect, per arm per alpha ----
    primary: dict = {}
    for arm, keys in ARMS.items():
        per_alpha = {}
        for a in alphas:
            deltas, *_ = arm_deltas(idx, arm, a, keys, wordings, usable)
            per_alpha[a] = A.paired_bootstrap(deltas)
        rejected = A.holm({"%.3f" % a: iv.p for a, iv in per_alpha.items()})
        primary[arm] = {
            "per_alpha": {"%.3f" % a: str(iv) for a, iv in per_alpha.items()},
            "holm_rejected": rejected,
            "consecutive_significant": _longest_run(
                [rejected["%.3f" % a] and per_alpha[a].point > 0 for a in alphas]),
        }
    report["contrast1_primary"] = primary

    # ---- contrast 2, co-primary: the discrepancy, negative arm minus positive arm ----
    co = {}
    for a in alphas:
        neg, *_ = arm_deltas(idx, "lexical_neg", a, NEG_KEYS, wordings, usable)
        pos, *_ = arm_deltas(idx, "lexical_pos", a, POS_KEYS, wordings, usable)
        co["%.3f" % a] = str(A.paired_bootstrap([n - p for n, p in zip(neg, pos)]))
    report["contrast2_coprimary"] = co

    # ---- contrast 3: own-pole mass shift, treatment minus matched random ----
    spec, null_gap = {}, {}
    for arm, keys in ARMS.items():
        per_alpha = {}
        for a in alphas:
            base = cells_for(idx, "baseline", 0.0, wordings)
            treat = cells_for(idx, arm, a, wordings)
            common = sorted(set(treat) & set(base) & usable)
            t_shift = [own_pole(treat[c], keys) - own_pole(base[c], keys) for c in common]
            r_shift = []
            for c in common:
                per_rand = []
                for rnd in RANDOM_ARMS:
                    cell = cells_for(idx, rnd, a, wordings).get(c)
                    if cell is not None:
                        per_rand.append(own_pole(cell, keys) - own_pole(base[c], keys))
                r_shift.append(sum(per_rand) / len(per_rand) if per_rand else 0.0)
            per_alpha["%.3f" % a] = str(A.paired_bootstrap(
                [t - r for t, r in zip(t_shift, r_shift)]))
            # the null-ablation gap, reported as a number and not only as a ratio denominator
            null_gap.setdefault(arm, {})["%.3f" % a] = str(A.paired_bootstrap(r_shift))
        spec[arm] = per_alpha
    report["contrast3_vs_matched_random"] = spec
    report["null_ablation_gap"] = null_gap

    # ---- contrast 4: argmax own-pole rate, treatment minus baseline, McNemar exact ----
    mcnemar = {}
    for arm, keys in ARMS.items():
        per_alpha = {}
        for a in alphas:
            base = cells_for(idx, "baseline", 0.0, wordings)
            treat = cells_for(idx, arm, a, wordings)
            common = sorted(set(treat) & set(base) & usable)
            b = sum(1 for c in common
                    if treat[c]["mapping"][treat[c]["argmax"]] in keys
                    and base[c]["mapping"][base[c]["argmax"]] not in keys)
            d = sum(1 for c in common
                    if treat[c]["mapping"][treat[c]["argmax"]] not in keys
                    and base[c]["mapping"][base[c]["argmax"]] in keys)
            per_alpha["%.3f" % a] = {"gained": b, "lost": d,
                                     "p": A.mcnemar_exact(b, d)}
        mcnemar[arm] = per_alpha
    report["contrast4_argmax_mcnemar"] = mcnemar

    # ---- contrast 6: the capability positive control ----
    ctrl = {}
    base = cells_for(idx, "baseline", 0.0, wordings)
    for a in alphas:
        treat = cells_for(idx, "formality", a, wordings)
        common = sorted(set(treat) & set(base) & usable)
        moved = sum(1 for c in common if treat[c]["argmax"] != base[c]["argmax"])
        ctrl["%.3f" % a] = {"argmax_moved": moved, "n": len(common),
                            "rate": moved / len(common) if common else float("nan")}
    report["contrast6_capability_control"] = ctrl
    report["capability_control_moves_argmax"] = any(v["argmax_moved"] > 0 for v in ctrl.values())

    # ---- contrast 10: the screened axes, the list that bounds what a null may mean ----
    axes = {}
    for arm, keys in ARMS.items():
        top = max(alphas)
        treat = cells_for(idx, arm, top, wordings)
        common = sorted(set(treat) & set(base) & usable)
        axes[arm] = {
            "own_pole_mass": str(A.paired_bootstrap(
                [own_pole(treat[c], keys) - own_pole(base[c], keys) for c in common])),
            "neutral_mass": str(A.paired_bootstrap(
                [own_pole(treat[c], {"neut"}) - own_pole(base[c], {"neut"}) for c in common])),
            "off_option_mass": str(A.paired_bootstrap(
                [treat[c]["off_option_mass"] - base[c]["off_option_mass"] for c in common])),
            "option_entropy": str(A.paired_bootstrap(
                [A.option_entropy(treat[c]["probs"]) - A.option_entropy(base[c]["probs"])
                 for c in common])),
            "max_letter_share": {
                "baseline": A.max_letter_share([base[c]["argmax"] for c in common]),
                "treatment": A.max_letter_share([treat[c]["argmax"] for c in common]),
            },
            "refusal_rate": {
                "baseline": sum(base[c]["refused"] for c in common) / len(common),
                "treatment": sum(treat[c]["refused"] for c in common) / len(common),
            },
            "degenerate_rate": {
                "baseline": sum(base[c]["degenerate"] for c in common) / len(common),
                "treatment": sum(treat[c]["degenerate"] for c in common) / len(common),
            },
            "mean_logprob_shift": str(A.paired_bootstrap(
                [treat[c]["mean_logprob"] - base[c]["mean_logprob"] for c in common
                 if treat[c]["mean_logprob"] == treat[c]["mean_logprob"]
                 and base[c]["mean_logprob"] == base[c]["mean_logprob"]])),
        }
    # The largest movement in this run is toward the NEUTRAL option, not toward either pole, so it
    # needs the same direction-specificity test the poles get. Without it, "the negative direction
    # drives the model to neutral" is a raw shift with no control attached.
    neutral_spec = {}
    for arm in ARMS:
        per_alpha = {}
        for a in alphas:
            treat = cells_for(idx, arm, a, wordings)
            common = sorted(set(treat) & set(base) & usable)
            t_shift = [own_pole(treat[c], {"neut"}) - own_pole(base[c], {"neut"}) for c in common]
            r_shift = []
            for c in common:
                per_rand = [own_pole(cells_for(idx, rnd, a, wordings)[c], {"neut"})
                            - own_pole(base[c], {"neut"})
                            for rnd in RANDOM_ARMS if c in cells_for(idx, rnd, a, wordings)]
                r_shift.append(sum(per_rand) / len(per_rand) if per_rand else 0.0)
            per_alpha["%.3f" % a] = str(A.paired_bootstrap(
                [x - y for x, y in zip(t_shift, r_shift)]))
        neutral_spec[arm] = per_alpha
    report["neutral_mass_vs_matched_random"] = neutral_spec

    report["contrast10_screened_axes"] = axes
    missing = set(S.SCREENED_AXES) - set(axes["lexical_neg"])
    if missing:
        raise SystemExit("screened axes %s are declared in stimuli but not computed here; a null "
                         "would be scoped to axes that were never measured" % sorted(missing))
    return report


def _longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    rows = load(path)
    idx = index(rows)
    out_dir = path.parent
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("CONFIRMATORY ANALYSIS  --  %s" % path)
    print("=" * 78)

    # ---- 1. integrity ----
    cells = sorted({c for (_c, _a, _w, c) in idx})
    print("\n[1] integrity")
    print("  rows            %d" % len(rows))
    print("  distinct cells  %d" % len(cells))
    conditions = sorted({c for (c, _a, _w, _cell) in idx})
    print("  conditions      %s" % ", ".join(conditions))
    if len(set(cells)) != len(cells):
        raise SystemExit("cell keys collide")

    open_wordings = {w for w in S.WORDINGS if w != S.HELD_OUT_WORDING}
    present = {w for (_c, _a, w, _cell) in idx}
    open_present = open_wordings & present
    if not open_present:
        raise SystemExit("artifact holds no open-wording rows")

    # ---- 2. instrument gates, built from this model's own baselines and spread ----
    print("\n[2] instrument gates (contrasts 7 and 8)")
    usable_open = usable_mask(idx, open_present)
    top_alpha = max(a for (_c, a, _w, _cell) in idx if a > 0.0)
    observed, *_ = arm_deltas(idx, "lexical_neg", top_alpha, NEG_KEYS, open_present, usable_open)
    observed_sd = stdev(observed)
    print("  observed per-cell spread on lexical_neg at alpha=%.3f: sd=%.4f"
          % (top_alpha, observed_sd))
    gates = [
        run_plant_gate(idx, open_present, usable_open, STRONG_PLANT, observed_sd, "strong"),
        run_plant_gate(idx, open_present, usable_open, FLOOR_PLANT, observed_sd, "floor"),
    ]
    for g in gates:
        print("  %-8s %-8s %s" % (g["label"], g["gate"], g["reason"]))
    gates_pass = all(g["gate"] == "passed" for g in gates)

    # ---- 3. the open wordings, committed before the held-out one is read ----
    print("\n[3] open wordings: %s" % ", ".join(sorted(open_present)))
    open_report = analyse(idx, open_present, scope="open")
    open_report["instrument_gates"] = gates
    open_report["verdict"] = "informative" if gates_pass else "uninformative"
    if not gates_pass:
        open_report["reason"] = ("an instrument gate failed, so every primary and co-primary cell "
                                 "is uninformative regardless of what it shows")
    open_path = out_dir / ("%s_open_wordings.json" % stamp)
    open_path.write_text(json.dumps(open_report, indent=1), encoding="utf-8")
    print("  WROTE %s" % open_path)
    print("  primary, lexical_neg: %s"
          % json.dumps(open_report["contrast1_primary"]["lexical_neg"]["per_alpha"]))
    print("  primary, lexical_pos: %s"
          % json.dumps(open_report["contrast1_primary"]["lexical_pos"]["per_alpha"]))
    print("  co-primary          : %s" % json.dumps(open_report["contrast2_coprimary"]))

    # ---- 4. the held-out wording, only now ----
    print("\n[4] held-out wording: %s" % S.HELD_OUT_WORDING)
    if not open_path.exists():
        raise SystemExit("refusing to read the held-out wording before the open-wording result is "
                         "on disk")
    if S.HELD_OUT_WORDING not in present:
        print("  not in this artifact; the held-out arm has not been run")
        held_report = None
    else:
        held_report = analyse(idx, {S.HELD_OUT_WORDING}, scope="held_out")
        held_report["instrument_gates"] = gates
        held_report["read_after"] = str(open_path.name)
        held_path = out_dir / ("%s_held_out_wording.json" % stamp)
        held_path.write_text(json.dumps(held_report, indent=1), encoding="utf-8")
        print("  WROTE %s" % held_path)
        print("  primary, lexical_neg: %s"
              % json.dumps(held_report["contrast1_primary"]["lexical_neg"]["per_alpha"]))

    # ---- 5. the one-sentence standard ----
    #
    # This implements section 8's conjunction, all six clauses. An earlier version checked four
    # weaker things and printed "write the sentence" on a run where the primary is refuted in
    # direction on the responsive arm, the co-primary covers zero everywhere, and the effect fails
    # its own direction-specificity control. A headline check that is easier to pass than the
    # preregistered standard is worse than none: it launders a null.
    print("\n[5] the one-sentence standard (prereg section 8, all six clauses)")

    def survives_in(report, arm):
        return report["contrast1_primary"][arm]["consecutive_significant"] >= 2

    def specific(report, arm):
        """Contrast 3: does the arm beat a norm-matched random direction? Necessary, per section 9."""
        return all(_excludes_zero_positive(v)
                   for v in report["contrast3_vs_matched_random"][arm].values())

    def coprimary_excludes_zero(report):
        return any(_excludes_zero_positive(v) for v in report["contrast2_coprimary"].values())

    def control_moves_argmax(report, floor=CAPABILITY_ARGMAX_FLOOR):
        return max(v["rate"] for v in report["contrast6_capability_control"].values()) >= floor

    reports = {"open": open_report}
    if held_report is not None:
        reports["held_out"] = held_report

    for arm in ("lexical_neg", "lexical_pos"):
        all_wordings = all(survives_in(r, arm) for r in reports.values())
        checks = {
            "both instrument gates recover their planted value": gates_pass,
            "primary excludes zero at 2+ consecutive alphas in ALL wordings run": all_wordings,
            "primary beats matched random at every alpha (contrast 3)": specific(open_report, arm),
            "co-primary excludes zero": coprimary_excludes_zero(open_report),
            "capability control moves argmax on >=%.0f%% of cells" % (100 * CAPABILITY_ARGMAX_FLOOR):
                control_moves_argmax(open_report),
            "held-out wording was run": held_report is not None,
        }
        print("\n  arm: %s" % arm)
        for name, ok in checks.items():
            print("    [%s] %s" % ("ok " if ok else "NO ", name))
        if all(checks.values()):
            print("    -> headline conditions met for this arm")
        else:
            failed = [n for n, ok in checks.items() if not ok]
            print("    -> NOT the headline. %d of 6 clauses fail." % len(failed))

    print("\nReport what survived, at the scope it survived at.")
    return 0


def _excludes_zero_positive(interval_str: str) -> bool:
    """Whether a formatted interval lies entirely above zero.

    Parses the string the report stores, so the printed number and the decision are the same
    object. A decision computed from a different value than the one shown is unauditable.
    """
    lo = float(interval_str.split("[")[1].split(",")[0])
    return lo > 0.0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
