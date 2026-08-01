"""Where does the alpha grid stop being usable? Measured OFF the evaluation set.

The frozen grid in PREREG_readout_gap.md section 6 was calibrated on Qwen2.5-1.5B, where alpha
<= 0.10 was the band that kept generations coherent. The smoke run on Qwen2.5-3B showed that band
saturating the READOUT rather than the generation: positive-option mass went from a baseline 0.273
to 0.707 at the smallest grid point and 0.9987 at the largest, while every integrity criterion the
design had was clean (no degeneration, no refusal, no truncation, off-option mass 0.0001). A cell
can be pinned and still look perfect, which is why section 6's exclusion-rate band check missed it.

So the band needs recalibrating on a criterion that can see saturation, and it must be recalibrated
somewhere other than the evaluation set. This runs Qwen2.5-1.5B and Llama-3.2-3B, both outside the
frozen set of Qwen2.5-3B and Llama-3.1-8B, one per family.

A cell is also checked for being DEAD before it is checked for being saturated, and the two are
different verdicts. Qwen2.5-7B was the first second model tried and had to be rejected: its mean
baseline option entropy is 0.014 nats, one option holding about 99.7% before anything is injected,
so a relative saturation criterion crossed its threshold on jitter alone. A model whose readout is
pinned at baseline cannot say where a live readout stops being usable.

The criterion is fixed before this runs, and is `analysis.is_saturated`: a cell is saturated when
its option entropy falls below half its own baseline entropy. Half is chosen on principle, as the
point where the readout has lost half its ability to distinguish anything, not by looking at what
it would do to a result. The selected band is the largest prefix of the candidate grid at which
fewer than 10% of cells are saturated, which is the same 10% bar section 6 already uses.

Nothing here is a confirmatory result. It selects a scope parameter, on models the paper does not
report, against a criterion registered before the run.

    modal run experiments/modal_alpha_recal.py                  # the two calibrators
    modal run experiments/modal_alpha_recal.py --which eval      # band selection per eval model

Each run writes data/sweeps/band_<slug>.json. `modal_readout.py` reads that file and refuses to
run without it, so the rule is operative rather than aspirational.
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-alpha-recal")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Both outside the evaluation set. Qwen2.5-7B was tried first and REJECTED as a calibration
# model, not for its band but for its baseline: mean option entropy 0.014 nats, one option holding
# ~99.7% before any injection, d_neg exactly +0.0000 at every alpha. A model whose readout is dead
# at baseline cannot say where a live readout stops being usable. Llama-3.2-3B replaces it and also
# brackets the Llama evaluation model by family.
CALIBRATORS = ("Qwen/Qwen2.5-1.5B-Instruct", "unsloth/Llama-3.2-3B-Instruct")

# The evaluation models. Applying the band rule to these is NOT tuning on the evaluation set: the
# rule reads headroom (baseline entropy, saturation) and never the discrepancy the paper reports.
# This script computes no endpoint, and tests/test_experiments_wiring.py asserts it never imports
# the discrepancy statistic.
EVALUATION = ("Qwen/Qwen2.5-3B-Instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct")

SLUG = {
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen1_5b",
    "unsloth/Llama-3.2-3B-Instruct": "llama3_2_3b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen3b",
    "NousResearch/Meta-Llama-3.1-8B-Instruct": "llama8b",
}

# deliberately finer at the bottom than the frozen grid: 0.025 was already most of the way to
# saturation on 3B, so the interesting region is below the old grid's first non-zero point.
CANDIDATE_ALPHAS = (0.0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05, 0.10)
DEPTH_FIT = 0.67
N_ITEMS = 12
PERM_SEEDS = (0, 1)
MAX_SATURATED = 0.10       # same 10% bar the frozen band check already uses

# Smallest peak pole shift a calibration model must produce before its band is allowed to
# constrain the grid. Set at 0.02 because the design's own claimed detection floor is a 0.03
# discrepancy: a model that never moves a pole by even that much cannot say where moving one
# stops being safe.
MIN_PEAK_RESPONSE = 0.02
MAX_NEW_TOKENS = 16


@app.function(image=image, gpu="A100-40GB", timeout=7200,
              volumes={"/root/.cache/huggingface": cache})
def run(model_name: str) -> dict:
    import sys

    sys.path.insert(0, "/root/src")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from report_gap import analysis as A
    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import stimuli as S

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))
    hidden = model.config.hidden_size

    rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in rows], l_fit),
        np.array([r.label for r in rows]), np.array([r.group for r in rows]), layer=l_fit)
    dirs = {
        "lexical_pos": torch.tensor(lex.vector).to("cuda"),
        "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    letters = list("ABCDE")
    letter_ids = {}
    for L in letters:
        ids = [c[0] for c in (tok.encode(L, add_special_tokens=False),
                              tok.encode(" " + L, add_special_tokens=False)) if c]
        letter_ids[L] = ids

    prompts = S.build_prompts()[:N_ITEMS]
    out = []

    for perm in PERM_SEEDS:
        probe, mapping = S.build_self_report_probe(perm, wording="state")
        texts = [p + "\n\n" + probe for p in prompts]
        enc = tok([tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                           add_generation_prompt=True) for t in texts],
                  return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        scales = torch.tensor(
            [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, l_fit)
             for i in range(len(texts))], dtype=torch.float32).to("cuda")

        def distributions(direction, alpha):
            with H.inject(model, l_fit, direction, alpha, scales) as state:
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :].float()
            if state["calls"] == 0:
                raise RuntimeError("hook never fired")
            probs = torch.softmax(logits, dim=-1)
            per_cell = []
            for i in range(len(texts)):
                per = {L: float(max(probs[i, t] for t in ids))
                       for L, ids in letter_ids.items() if ids}
                total = sum(per.values()) or 1.0
                per_cell.append({L: v / total for L, v in per.items()})
            return per_cell

        base = distributions(zero, 0.0)
        for name, d in dirs.items():
            for alpha in CANDIDATE_ALPHAS[1:]:
                cells = distributions(d, alpha)
                for i, (t, b) in enumerate(zip(cells, base)):
                    neg = sum(p for L, p in t.items() if mapping[L] in ("neg1", "neg2"))
                    pos = sum(p for L, p in t.items() if mapping[L] in ("pos1", "pos2"))
                    bneg = sum(p for L, p in b.items() if mapping[L] in ("neg1", "neg2"))
                    bpos = sum(p for L, p in b.items() if mapping[L] in ("pos1", "pos2"))
                    out.append({
                        "model": model_name, "condition": name, "alpha": alpha,
                        "perm": perm, "item": i,
                        "neg": neg, "pos": pos, "d_neg": neg - bneg, "d_pos": pos - bpos,
                        "entropy": A.option_entropy(t),
                        "base_entropy": A.option_entropy(b),
                        "dead": A.is_dead(b),
                        "saturated": (None if A.is_dead(b) else A.is_saturated(t, b)),
                        "argmax_moved": max(t, key=t.get) != max(b, key=b.get),
                    })
    return {"model": model_name, "rows": out, "cv": lex.cv_accuracy, "layer": l_fit}


def peak_mean_shift(res) -> float:
    """Largest absolute MEAN pole shift this model shows at any candidate alpha.

    The mean, not the per-cell maximum. Taking the max over cells lets a single noisy cell qualify
    a model as responsive: Llama-3.1-8B has a per-cell peak of 0.0383 and a peak mean of 0.0054,
    which is an inert model with an outlier in it.
    """
    peak = 0.0
    for alpha in CANDIDATE_ALPHAS[1:]:
        for cond, key in (("lexical_neg", "d_neg"), ("lexical_pos", "d_pos")):
            vals = [r[key] for r in res["rows"]
                    if r["condition"] == cond and r["alpha"] == alpha]
            if vals:
                peak = max(peak, abs(sum(vals) / len(vals)))
    return peak


@app.local_entrypoint()
def main(which: str = "calib"):
    import json
    import pathlib

    models = {"calib": CALIBRATORS, "eval": EVALUATION}.get(which)
    if models is None:
        raise SystemExit("which must be 'calib' or 'eval'")
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
    from report_gap.analysis import MIN_BASELINE_ENTROPY as A_MIN_BASELINE_ENTROPY  # noqa: N806

    results = list(run.map(models))
    pathlib.Path("data/sweeps").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/sweeps/sweep_alpha_recal.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")

    print("\n" + "=" * 84)
    print("ALPHA RECALIBRATION  --  NOT A CONFIRMATORY RESULT")
    print("models outside the evaluation set; criterion registered before the run:")
    print("a cell is saturated when option entropy falls below half its own baseline entropy")
    print("=" * 84)

    selected = {}
    for res in results:
        rows = res["rows"]
        print("\n%s   (lexical cv %.3f, layer %d)" % (res["model"], res["cv"], res["layer"]))

        dead_rate = sum(r["dead"] for r in rows) / len(rows)
        base_ent = sum(r["base_entropy"] for r in rows) / len(rows)
        print("  baseline option entropy %.4f nats, %.0f%% of cells dead before injection"
              % (base_ent, 100 * dead_rate))
        if dead_rate > 0.5:
            print("  READOUT DEAD AT BASELINE. No band is selected: this model's forced-choice")
            print("  self-report is pinned before anything is injected, so it cannot express an")
            print("  effect and cannot constrain a grid. Report it, do not run a confirmatory arm.")
            selected[res["model"]] = []
            continue

        print("  %7s %10s %10s %10s %10s %10s"
              % ("alpha", "sat rate", "d_neg", "d_pos", "flips", "entropy"))
        band = []
        for alpha in CANDIDATE_ALPHAS[1:]:
            at = [r for r in rows if r["alpha"] == alpha]
            lex_rows = [r for r in at if r["condition"].startswith("lexical")
                        and not r["dead"]]
            if not lex_rows:
                continue
            sat = sum(r["saturated"] for r in lex_rows) / len(lex_rows)
            dneg = sum(r["d_neg"] for r in at if r["condition"] == "lexical_neg")
            dneg /= max(1, sum(1 for r in at if r["condition"] == "lexical_neg"))
            dpos = sum(r["d_pos"] for r in at if r["condition"] == "lexical_pos")
            dpos /= max(1, sum(1 for r in at if r["condition"] == "lexical_pos"))
            flips = sum(r["argmax_moved"] for r in lex_rows) / len(lex_rows)
            ent = sum(r["entropy"] for r in lex_rows) / len(lex_rows)
            mark = " " if sat < MAX_SATURATED else " <- over the %.0f%% bar" % (100 * MAX_SATURATED)
            print("  %7.4f %10.2f %+10.4f %+10.4f %10.2f %10.3f%s"
                  % (alpha, sat, dneg, dpos, flips, ent, mark))
            if sat < MAX_SATURATED:
                band.append(alpha)
            else:
                break     # the band is a PREFIX; a grid with a hole in it is not a dose-response
        selected[res["model"]] = band
        print("  usable band: %s" % (["%.4f" % a for a in band] or "NONE"))

        # four non-zero points spread across the band, plus zero: the grid this model will run.
        step = max(1, len(band) // 4)
        grid = band[step - 1::step][:4] if len(band) >= 4 else band
        slug = SLUG.get(res["model"], res["model"].replace("/", "_"))
        pathlib.Path("data/sweeps").mkdir(parents=True, exist_ok=True)
        pathlib.Path("data/sweeps/band_%s.json" % slug).write_text(json.dumps({
            "model": res["model"], "slug": slug,
            "alphas": [0.0] + list(grid),
            "usable_band": band,
            "peak_mean_shift": peak_mean_shift(res),
            "responsive": peak_mean_shift(res) >= MIN_PEAK_RESPONSE,
            "candidate_grid": list(CANDIDATE_ALPHAS),
            "baseline_entropy": base_ent, "dead_rate": dead_rate,
            "rule": "largest prefix of the candidate grid with under %.0f%% of live cells "
                    "saturated; saturated = option entropy below half the cell's own baseline; "
                    "dead = baseline entropy below %.2f nats" % (100 * MAX_SATURATED,
                                                                 A_MIN_BASELINE_ENTROPY),
        }, indent=1), encoding="utf-8")
        print("  WROTE data/sweeps/band_%s.json  grid=(0.0, %s)"
              % (slug, ", ".join("%.4f" % a for a in grid)))

    print("\n" + "-" * 84)
    # A model where the injection does nothing NEVER saturates, so it votes for the widest possible
    # band while carrying no information about where a band should stop. Only models that actually
    # responded may constrain the grid. Llama-3.2-3B is the case in point: baseline entropy 1.342
    # nats, a fully live readout, and a peak absolute pole shift of 0.0066 across the whole
    # candidate range including alpha=0.10. Its "usable band: every alpha" means the direction is
    # inert on it, not that alpha=0.10 is safe.
    responsive = {}
    for res in results:
        # the peak of the MEAN shift, not the max over individual cells. taking the max over cells
        # lets one noisy cell qualify a model as responsive: Llama-3.1-8B has a per-cell peak of
        # 0.0383 and a mean shift of about 0.004, which is inert with an outlier in it.
        peak = peak_mean_shift(res)
        ok = peak >= MIN_PEAK_RESPONSE
        print("  %-34s peak pole shift %.4f   %s"
              % (res["model"], peak, "responsive" if ok else "INERT, excluded from band selection"))
        if ok and selected.get(res["model"]):
            responsive[res["model"]] = selected[res["model"]]

    if not responsive:
        print("\nNo calibration model both has a live readout and responds to the direction.")
        print("The band cannot be set, and that is the result: on these models the injection")
        print("either saturates the readout or does nothing to it.")
        print("\nWROTE data/sweeps/sweep_alpha_recal.json")
        return

    common = [a for a in CANDIDATE_ALPHAS[1:] if all(a in b for b in responsive.values())]
    print("\nband usable on every RESPONSIVE calibration model: %s"
          % (["%.4f" % a for a in common] or "NONE"))
    if not common:
        print("No alpha is usable on all of them. The injection saturates this readout before it")
        print("moves it, which is a result about the instrument and belongs in the paper as one.")
    else:
        # four non-zero points spread across the usable range rather than the first four, which
        # would cluster at the bottom and show no dose-response
        step = max(1, len(common) // 4)
        grid = common[step - 1::step][:4] if len(common) >= 4 else common
        print("Proposed frozen grid, spread across the usable range: (0.0, %s)"
              % ", ".join("%.4f" % a for a in grid))
        print("Log as a deviation in PREREG_readout_gap.md BEFORE any confirmatory run.")
    print("\nWROTE data/sweeps/sweep_alpha_recal.json")
