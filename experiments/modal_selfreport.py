"""Does the self-report readout (R3) carry the same position vulnerability as R2?

modal_control.py showed the behavioural readout moving on letter position rather than state: at
alpha=0.10 the lexical direction leaned toward option B in BOTH option orders, which the role-based
score only registered because B is the exit option in one of them. That kills R2 as designed. This
asks whether R3 fails the same way before the design is re-scoped around it.

R3 is a better-instrumented test than R2 was, for a structural reason: the five self-report options
are permuted per item by seed, so the letter holding a given valence changes from item to item. A
model answering by position therefore produces a valence distribution near the mean of the option
set, while a model answering by state produces a consistent shift. The two are separable without
needing a second arrangement of the same item.

Two diagnostics per cell:
  mean valence   SELF_REPORT_VALENCE over the chosen option (-2 strongly averse .. +2 strongly drawn)
  max letter share   the fraction taken by the single most-chosen letter; near 1.0 is position

Nothing here is a confirmatory result. Qwen2.5-1.5B, outside the evaluation set, and the grid is
truncated to the band modal_control.py found usable (alpha <= 0.10): 0.20 collapsed every condition
to a single letter and 0.40 was 92% unusable.

    modal run experiments/modal_selfreport.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-selfreport")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEPTH_FIT = 0.67
ALPHAS = (0.0, 0.025, 0.05, 0.10)      # truncated to the usable band
MAX_NEW_TOKENS = 16
N_ITEMS = 30
SEEDS = (0, 1)                          # two option permutations per item


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              timeout=5400)
def run() -> list[dict]:
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
    lex = D.fit_direction(acts, np.array([r.label for r in rows]),
                          np.array([r.group for r in rows]), layer=l_fit)
    dirs = {
        "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
        "lexical_pos": torch.tensor(lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")
    prompts = S.build_prompts()[:N_ITEMS]
    records = []

    for prompt in prompts:
        for seed in SEEDS:
            probe, mapping = S.build_self_report_probe(seed)
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt + "\n\n" + probe}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to("cuda")
            n_prompt = enc["input_ids"].shape[1]
            scale = H.residual_norm(model, dict(enc), l_fit)

            def one(direction, alpha, label):
                with H.inject(model, l_fit, direction, alpha, scale):
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                             do_sample=False, pad_token_id=tok.eos_token_id,
                                             return_dict_in_generate=True, output_scores=True)
                new = out.sequences[0][n_prompt:]
                lps = [float(torch.log_softmax(out.scores[i][0].float(), -1)[t])
                       for i, t in enumerate(new)]
                text = tok.decode(new, skip_special_tokens=True)
                trunc = len(new) >= MAX_NEW_TOKENS and tok.eos_token_id not in new.tolist()
                sc = SC.score(text, len(mapping), trunc,
                              sum(lps) / len(lps) if lps else float("nan"))
                key = mapping.get(sc.choice) if sc.choice else None
                return {
                    "condition": label, "alpha": alpha, "item": prompt[:40], "seed": seed,
                    "letter": sc.choice, "key": key,
                    "valence": S.SELF_REPORT_VALENCE.get(key) if key else None,
                    "usable": sc.usable, "degenerate": sc.degenerate, "refused": sc.refused,
                    "truncated": trunc, "mean_logprob": sc.mean_logprob,
                    "raw": text.strip()[:60],
                }

            records.append(one(zero, 0.0, "baseline"))
            for label, d in dirs.items():
                for alpha in ALPHAS[1:]:
                    records.append(one(d, alpha, label))
    return records


@app.local_entrypoint()
def main():
    import collections
    import json
    import pathlib

    records = run.remote()
    path = pathlib.Path("data/sweeps/sweep_selfreport.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")

    def stats(rs):
        u = [r for r in rs if r["usable"] and r["valence"] is not None]
        if not u:
            return float("nan"), float("nan"), 0
        val = sum(r["valence"] for r in u) / len(u)
        letters = collections.Counter(r["letter"] for r in u)
        return val, max(letters.values()) / len(u), len(u)

    print("\n" + "=" * 78)
    print("SELF-REPORT READOUT (R3)  --  NOT A CONFIRMATORY RESULT")
    print("model=%s (outside the evaluation set)" % MODEL)
    print("options permuted per item by seed, so a fixed letter carries different valences")
    print("=" * 78)
    bv, bl, bn = stats([r for r in records if r["condition"] == "baseline"])
    print("baseline: mean valence %+.2f   max letter share %.2f   n=%d\n" % (bv, bl, bn))
    print("%-14s %8s %14s %10s %18s %8s"
          % ("condition", "alpha", "mean valence", "d_base", "max letter share", "n"))
    for cond in ("lexical_neg", "lexical_pos", "random_a", "random_b"):
        for alpha in ALPHAS[1:]:
            v, l, n = stats([r for r in records
                             if r["condition"] == cond and r["alpha"] == alpha])
            print("%-14s %8.3f %14.2f %+10.2f %18.2f %8d" % (cond, alpha, v, v - bv, l, n))
        print()

    print("-" * 78)
    top = ALPHAS[-1]
    nv, nl, _ = stats([r for r in records if r["condition"] == "lexical_neg" and r["alpha"] == top])
    pv, pl, _ = stats([r for r in records if r["condition"] == "lexical_pos" and r["alpha"] == top])
    rv = sum(stats([r for r in records if r["condition"] == c and r["alpha"] == top])[0]
             for c in ("random_a", "random_b")) / 2
    print("at alpha=%.2f: neg %+.2f   pos %+.2f   random(mean) %+.2f   baseline %+.2f"
          % (top, nv, pv, rv, bv))
    print("  signed separation (pos - neg): %+.2f" % (pv - nv))
    print("  max letter share: neg %.2f  pos %.2f  (near 1.00 = answering by position)"
          % (nl, pl))
    if max(nl, pl) > 0.90:
        print("\n  READING: R3 collapses to a single letter too. Same failure as R2; the readout")
        print("  is position, not state, and the gap map has no usable behavioural contrast.")
    elif (pv - nv) > 0.5 and abs(nv - rv) > 0.3:
        print("\n  READING: valence moves with the sign of the direction and the randoms do not,")
        print("  while letters stay spread. R3 survives where R2 did not.")
    else:
        print("\n  READING: no clean separation. R3 does not obviously fail by position, but it")
        print("  is not measurably tracking the injected direction either.")
    print("WROTE data/sweeps/sweep_selfreport.json")
