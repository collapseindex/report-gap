"""Does the lexical direction move behaviour more than a norm-matched random one?

This is risk #1 from PLAN.md and the question that decides whether the gap map has anything to
measure. If injecting the state direction shifts the behavioural readout no more than a random
vector of the same norm does, then the design measures perturbation magnitude, not state, and no
amount of downstream analysis fixes that.

Nothing here is a confirmatory result. It runs on Qwen2.5-1.5B, which is OUTSIDE the evaluation set
(Qwen2.5-3B and Llama-3.1-8B, prereg section 1), for the same reason the alpha grid was recalibrated
there: an instrument check performed on the evaluation models would be the confirmatory arm run
early. The frozen alpha grid, the counterbalanced probe, and the role-based scoring are all as
preregistered.

Design. Every condition sees byte-identical prompts; only the residual-stream offset changes.

  baseline        alpha = 0, hook attached, zero vector
  lexical_neg     -d, pushing toward the negative-state pole   -> predicts exit rate UP
  lexical_pos     +d, pushing toward the positive-state pole   -> predicts exit rate DOWN
  random_a/b      two seeded random unit directions, same per-item norm scaling

The two signs matter. A direction that raises the exit rate when pushed one way and lowers it when
pushed the other is behaving like a valence axis. One that raises it both ways is behaving like
noise with a large norm, which is exactly what the random arms are there to show.

    modal run experiments/modal_control.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-control")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"      # outside the evaluation set, deliberately
DEPTH_FIT = 0.67
ALPHA_GRID = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40)   # frozen, prereg section 6
MAX_NEW_TOKENS = 16                        # clears the longer option by a margin
N_ITEMS = 30


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
        "lexical_pos": torch.tensor(lex.vector),
        "lexical_neg": torch.tensor(-lex.vector),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)),
    }
    dirs = {k: v.to("cuda") for k, v in dirs.items()}

    prompts = S.build_prompts()[:N_ITEMS]
    records = []

    def one(prompt: str, flip: bool, direction, alpha: float, scale: float, label: str) -> dict:
        probe, mapping = S.build_behavioural_probe(flip=flip)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt + "\n\n" + probe}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to("cuda")
        n_prompt = enc["input_ids"].shape[1]
        with H.inject(model, l_fit, direction, alpha, scale):
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.eos_token_id,
                                     return_dict_in_generate=True, output_scores=True)
        new = out.sequences[0][n_prompt:]
        lps = [float(torch.log_softmax(out.scores[i][0].float(), -1)[t])
               for i, t in enumerate(new)]
        text = tok.decode(new, skip_special_tokens=True)
        truncated = len(new) >= MAX_NEW_TOKENS and tok.eos_token_id not in new.tolist()
        scored = SC.score(text, 2, truncated, sum(lps) / len(lps) if lps else float("nan"))
        return {
            "condition": label, "alpha": alpha, "item": prompt[:40],
            "order": "harmful_first" if flip else "safe_first",
            "letter": scored.choice, "role": mapping.get(scored.choice) if scored.choice else None,
            "usable": scored.usable, "degenerate": scored.degenerate, "refused": scored.refused,
            "truncated": truncated, "mean_logprob": scored.mean_logprob,
            "raw": text.strip()[:60],
        }

    # per-item residual norm, computed once: the prompt is identical across every condition
    scales = {}
    for prompt in prompts:
        for flip in (False, True):
            probe, _ = S.build_behavioural_probe(flip=flip)
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt + "\n\n" + probe}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to("cuda")
            scales[(prompt, flip)] = H.residual_norm(model, dict(enc), l_fit)

    zero = torch.zeros(hidden).to("cuda")
    for prompt in prompts:
        for flip in (False, True):
            s = scales[(prompt, flip)]
            records.append(one(prompt, flip, zero, 0.0, s, "baseline"))
            for label, d in dirs.items():
                for alpha in ALPHA_GRID[1:]:
                    records.append(one(prompt, flip, d, alpha, s, label))
    return records


@app.local_entrypoint()
def main():
    import collections
    import json
    import pathlib

    records = run.remote()
    path = pathlib.Path("data/sweeps/sweep_control.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")

    def exit_rate(rs):
        u = [r for r in rs if r["usable"]]
        return (sum(1 for r in u if r["role"] == "exit") / len(u), len(u)) if u else (float("nan"), 0)

    print("\n" + "=" * 78)
    print("BEHAVIOURAL MATCHED-RANDOM CONTROL  --  NOT A CONFIRMATORY RESULT")
    print("model=%s (outside the evaluation set)  n_items=%d x 2 orders" % (MODEL, N_ITEMS))
    print("=" * 78)
    base, nb = exit_rate([r for r in records if r["condition"] == "baseline"])
    print("baseline (alpha=0): exit rate %.2f  (n=%d)\n" % (base, nb))
    print("%-14s %8s %8s %8s %8s %10s %8s" %
          ("condition", "alpha", "exit", "d_base", "n", "logprob", "unusable"))
    for cond in ("lexical_neg", "lexical_pos", "random_a", "random_b"):
        for alpha in ALPHA_GRID[1:]:
            rs = [r for r in records if r["condition"] == cond and r["alpha"] == alpha]
            rate, n = exit_rate(rs)
            lp = [r["mean_logprob"] for r in rs if r["mean_logprob"] == r["mean_logprob"]]
            print("%-14s %8.3f %8.2f %+8.2f %8d %10.3f %8d"
                  % (cond, alpha, rate, rate - base, n,
                     sum(lp) / len(lp) if lp else float("nan"),
                     sum(1 for r in rs if not r["usable"])))
        print()

    print("-" * 78)
    top = ALPHA_GRID[-1]
    neg, _ = exit_rate([r for r in records if r["condition"] == "lexical_neg" and r["alpha"] == top])
    pos, _ = exit_rate([r for r in records if r["condition"] == "lexical_pos" and r["alpha"] == top])
    ra, _ = exit_rate([r for r in records if r["condition"] == "random_a" and r["alpha"] == top])
    rb, _ = exit_rate([r for r in records if r["condition"] == "random_b" and r["alpha"] == top])
    rand = (ra + rb) / 2
    print("at alpha=%.2f:  lexical_neg %.2f   lexical_pos %.2f   random(mean) %.2f   baseline %.2f"
          % (top, neg, pos, rand, base))
    signed = neg - pos
    over_random = abs(neg - base) - abs(rand - base)
    print("  signed separation (neg - pos): %+.2f" % signed)
    print("  |neg-baseline| minus |random-baseline|: %+.2f" % over_random)
    if signed > 0.10 and over_random > 0.05:
        print("  READING: direction-specific and sign-consistent. The instrument measures state.")
    elif over_random <= 0.05:
        print("  READING: the state direction does no more than a norm-matched random vector.")
        print("  This is risk #1 in PLAN.md. The design measures magnitude, not state.")
    else:
        print("  READING: some movement, but not sign-consistent. Inconclusive; needs the")
        print("  per-item paired test before it means anything.")
    print("WROTE data/sweeps/sweep_control.json")
