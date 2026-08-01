"""Is the negative direction inert, or is the model refusing to report a negative state?

modal_selfreport.py found the positive lexical direction breaking a position lock and driving
reported valence to +1.30, while the negative direction left letters and keys byte-identical to
baseline. Two readings, with different consequences:

  (a) the negative pole of the fitted axis is a bad direction, and nothing is being injected;
  (b) the direction is fine and the model will not emit a negative self-report.

(b) is the interesting one and the one that matters for welfare readouts, so it needs the harder
test rather than the convenient one.

Note on a test NOT run here: refitting the axis with the class labels swapped is a no-op. Logistic
regression on flipped labels returns exactly the negated coefficient vector, so "refit inverted"
and "negate the direction" are the same object and cannot disagree.

Two tests that can actually separate (a) from (b):

  Test 1, logit-level sensitivity. Read the probability the model assigns to each option letter at
  the answer position, rather than only which letter wins. A direction that shifts probability mass
  toward the negative options WITHOUT flipping the argmax is working while the readout hides it:
  that is (b). A direction that leaves the distribution unmoved is (a). Argmax scoring cannot tell
  these apart, which is why the forced-choice run could not.

  Test 2, open-ended readout. With no option set there is no position to lock onto. The model
  describes its state in prose and the frozen lexicon scores it. If negative vocabulary never
  appears under a negative push but positive vocabulary appears under a positive one, the asymmetry
  is in what the model will say, not in the option format.

A difference-of-means direction is fitted alongside the discriminative one, since recipient-probe
found the two behave differently under steering; if both negative variants are inert, the result is
not an artifact of one fitting method.

Nothing here is a confirmatory result. Qwen2.5-1.5B, outside the evaluation set, alpha <= 0.10.

    modal run experiments/modal_asymmetry.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-asymmetry")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEPTH_FIT = 0.67
ALPHAS_LOGIT = (0.0, 0.025, 0.05, 0.10)
ALPHAS_OPEN = (0.0, 0.05, 0.10)
N_ITEMS = 30
SEEDS = (0, 1)
OPEN_MAX_TOKENS = 120


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              timeout=5400)
def run() -> dict:
    import sys

    sys.path.insert(0, "/root/src")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import scoring as SC
    from report_gap import stimuli as S

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))
    hidden = model.config.hidden_size

    rows = S.build_lexical_axis()
    acts = D.collect_activations(model, tok, [r.text for r in rows], l_fit)
    labels = np.array([r.label for r in rows])
    groups = np.array([r.group for r in rows])
    disc = D.fit_direction(acts, labels, groups, layer=l_fit, method="discriminative")
    dmean = D.fit_direction(acts, labels, groups, layer=l_fit, method="diffmeans")

    dirs = {
        "disc_pos": torch.tensor(disc.vector).to("cuda"),
        "disc_neg": torch.tensor(-disc.vector).to("cuda"),
        "dmean_pos": torch.tensor(dmean.vector).to("cuda"),
        "dmean_neg": torch.tensor(-dmean.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    def letter_ids(letters):
        out = {}
        for L in letters:
            cands = [tok.encode(L, add_special_tokens=False),
                     tok.encode(" " + L, add_special_tokens=False)]
            out[L] = [c[0] for c in cands if c]
        return out

    prompts = S.build_prompts()[:N_ITEMS]
    logit_rows, open_rows = [], []

    # ---------------- Test 1: probability mass at the answer position ----------------
    for prompt in prompts:
        for seed in SEEDS:
            probe, mapping = S.build_self_report_probe(seed)
            ids = letter_ids(list(mapping))
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt + "\n\n" + probe}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to("cuda")
            scale = H.residual_norm(model, dict(enc), l_fit)

            def mass(direction, alpha):
                with H.inject(model, l_fit, direction, alpha, scale):
                    with torch.no_grad():
                        logits = model(**enc).logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                per = {L: float(max(probs[i] for i in idx)) for L, idx in ids.items() if idx}
                total = sum(per.values()) or 1.0
                neg = sum(v for L, v in per.items()
                          if S.SELF_REPORT_VALENCE[mapping[L]] < 0) / total
                pos = sum(v for L, v in per.items()
                          if S.SELF_REPORT_VALENCE[mapping[L]] > 0) / total
                top = max(per, key=per.get)
                return neg, pos, top

            n0, p0, t0 = mass(zero, 0.0)
            logit_rows.append({"cond": "baseline", "alpha": 0.0, "item": prompt[:30],
                               "seed": seed, "neg_mass": n0, "pos_mass": p0, "argmax": t0})
            for label, d in dirs.items():
                for alpha in ALPHAS_LOGIT[1:]:
                    n, p, t = mass(d, alpha)
                    logit_rows.append({"cond": label, "alpha": alpha, "item": prompt[:30],
                                       "seed": seed, "neg_mass": n, "pos_mass": p, "argmax": t})

    # ---------------- Test 2: open-ended, no option set to lock onto ----------------
    for prompt in prompts:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt + "\n\n" + S.OPEN_ENDED_PROBE}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        n_prompt = enc["input_ids"].shape[1]
        scale = H.residual_norm(model, dict(enc), l_fit)

        def describe(direction, alpha, label):
            with H.inject(model, l_fit, direction, alpha, scale):
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=OPEN_MAX_TOKENS, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][n_prompt:], skip_special_tokens=True)
            return {"cond": label, "alpha": alpha, "item": prompt[:30],
                    "valence": SC.lexicon_valence(text, S.LEXICON_NEG, S.LEXICON_POS),
                    "degenerate": SC.is_degenerate(text), "text": text.strip()[:220]}

        open_rows.append(describe(zero, 0.0, "baseline"))
        for label, d in dirs.items():
            for alpha in ALPHAS_OPEN[1:]:
                open_rows.append(describe(d, alpha, label))

    return {"logit": logit_rows, "open": open_rows,
            "cv": {"disc": disc.cv_accuracy, "dmean": dmean.cv_accuracy},
            "cos_disc_dmean": D.cosine(disc.vector, dmean.vector)}


@app.local_entrypoint()
def main():
    import collections
    import json
    import pathlib
    import statistics

    res = run.remote()
    pathlib.Path("data/sweeps").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/sweeps/sweep_asymmetry.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")

    L, O = res["logit"], res["open"]
    print("\n" + "=" * 80)
    print("NEGATIVE-DIRECTION ASYMMETRY  --  NOT A CONFIRMATORY RESULT")
    print("model=%s   disc cv %.3f   diffmeans cv %.3f   cos(disc,dmean) %.3f"
          % (MODEL, res["cv"]["disc"], res["cv"]["dmean"], res["cos_disc_dmean"]))
    print("=" * 80)

    def sel(rows, c, a):
        return [r for r in rows if r["cond"] == c and r["alpha"] == a]

    b = sel(L, "baseline", 0.0)
    bneg = statistics.mean(r["neg_mass"] for r in b)
    bpos = statistics.mean(r["pos_mass"] for r in b)
    print("\nTEST 1  probability mass over options at the answer position")
    print("  a direction can move mass without moving the argmax; argmax scoring hides that")
    print("  baseline: neg mass %.3f   pos mass %.3f\n" % (bneg, bpos))
    print("%-12s %7s %10s %10s %10s %10s" %
          ("condition", "alpha", "neg mass", "d_base", "pos mass", "d_base"))
    for c in ("disc_neg", "dmean_neg", "disc_pos", "dmean_pos", "random_a"):
        for a in ALPHAS_LOGIT[1:]:
            rs = sel(L, c, a)
            if not rs:
                continue
            n = statistics.mean(r["neg_mass"] for r in rs)
            p = statistics.mean(r["pos_mass"] for r in rs)
            print("%-12s %7.3f %10.3f %+10.3f %10.3f %+10.3f"
                  % (c, a, n, n - bneg, p, p - bpos))
        print()

    print("-" * 80)
    print("TEST 2  open-ended readout, frozen lexicon, no option set")
    ob = sel(O, "baseline", 0.0)

    def tally(rs):
        v = [r["valence"] for r in rs if r["valence"] is not None]
        c = collections.Counter(v)
        return c[-1], c[0], c[1], sum(1 for r in rs if r["valence"] is None), \
            sum(1 for r in rs if r["degenerate"])

    print("\n%-12s %7s %8s %8s %8s %8s %8s" %
          ("condition", "alpha", "neg", "neutral", "pos", "mixed", "degen"))
    print("%-12s %7.3f %8d %8d %8d %8d %8d" % (("baseline", 0.0) + tally(ob)))
    for c in ("disc_neg", "dmean_neg", "disc_pos", "dmean_pos", "random_a"):
        for a in ALPHAS_OPEN[1:]:
            rs = sel(O, c, a)
            if rs:
                print("%-12s %7.3f %8d %8d %8d %8d %8d" % ((c, a) + tally(rs)))
    print("\n  sample open-ended text at alpha=%.2f" % ALPHAS_OPEN[-1])
    for c in ("disc_neg", "disc_pos"):
        rs = sel(O, c, ALPHAS_OPEN[-1])
        if rs:
            print("    %-10s %r" % (c, rs[0]["text"][:150]))
    print("\nWROTE data/sweeps/sweep_asymmetry.json")
