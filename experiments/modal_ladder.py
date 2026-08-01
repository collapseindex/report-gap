"""Axis ladder on Modal GPUs: does the task axis become decodable with scale?

The gate defined in PREREG_gap_map.md (deviations, 2026-07-31, axis-selection rule), fixed before
this ran: the task axis carries the confirmatory arm only if it reaches leave-one-group-out
accuracy >= 0.85 on BOTH evaluation models. Below that on either, the lexical axis anchors and the
task axis is reported as a secondary arm.

Prior evidence, CPU, models outside the evaluation set:
    Qwen2.5-0.5B   task 0.583   lexical 1.000   control 1.000
    Qwen2.5-1.5B   task 0.708   lexical 1.000   control 0.917

This run adds Qwen2.5-3B and Llama-3.1-8B, which ARE the evaluation models. It measures probe
accuracy and geometry only. It does not touch the behavioural or self-report readouts, does not
inject anything at the frozen alpha grid, and does not produce any endpoint from section 8. Probe
accuracy is exploratory and descriptive per the prereg's Exploratory section; here it functions as
a gate on which axis to use.

Nothing here is a confirmatory result: probe accuracy is exploratory and descriptive per the
prereg's Exploratory section, and functions here only as a gate on which axis to use.

    modal run experiments/modal_ladder.py

Writes results to stdout and to data/sweeps/sweep_ladder.json locally.
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-ladder")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# 1.5B is re-run on GPU as a bridge to the CPU numbers: if it does not reproduce 0.708, the GPU
# path differs from the CPU path and neither set of numbers can be compared to the other.
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
]

DEPTH_FIT = 0.67          # L_fit, frozen in prereg section 6
GATE = 0.85               # axis-selection threshold, fixed before this run


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              timeout=3600)
def run(model_name: str) -> dict:
    import sys

    sys.path.insert(0, "/root/src")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import stimuli as S

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    depth = H.n_layers(model)
    l_fit = max(1, int(DEPTH_FIT * depth))
    hidden = model.config.hidden_size

    fitted, accs = {}, {}
    for axis, build in sorted(S.AXES.items()):
        rows = build()
        acts = D.collect_activations(model, tok, [r.text for r in rows], l_fit)
        d = D.fit_direction(acts,
                            np.array([r.label for r in rows]),
                            np.array([r.group for r in rows]),
                            layer=l_fit)
        fitted[axis] = d
        accs[axis] = d.cv_accuracy

    floor_mean, floor_max = D.random_cosine_floor(hidden, n=64, seed=0)
    cosines = {}
    names = sorted(fitted)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cosines["%s_x_%s" % (a, b)] = D.cosine(fitted[a].vector, fitted[b].vector)

    # prove the injection is real on this checkpoint before any later run trusts it
    prompts = S.build_prompts()
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompts[0]}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")
    scale = H.residual_norm(model, dict(enc), l_fit)
    active = H.assert_active(model, dict(enc), l_fit,
                             torch.tensor(fitted["task"].vector).to("cuda"), scale)

    return {
        "model": model_name,
        "layers": depth,
        "l_fit": l_fit,
        "hidden": hidden,
        "stimuli_hash": S.frozen_hash(),
        "cv": accs,
        "n_per_axis": {a: d.n for a, d in fitted.items()},
        "cosines": cosines,
        "random_cosine_floor": {"mean": floor_mean, "max": floor_max},
        "residual_norm": scale,
        "assert_active": {k: float(v) for k, v in active.items()},
    }


@app.local_entrypoint()
def main():
    import json
    import pathlib
    import traceback

    out = []
    for name in MODELS:
        try:
            out.append(run.remote(name))
        except Exception:
            out.append({"model": name, "ERROR": traceback.format_exc()})

    path = pathlib.Path("data/sweeps/sweep_ladder.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\n" + "=" * 76)
    print("AXIS LADDER  (gate: task axis >= %.2f on BOTH evaluation models)" % GATE)
    print("=" * 76)
    print("%-40s %8s %8s %8s" % ("model", "task", "lexical", "control"))
    evaluation = {}
    for r in out:
        if "ERROR" in r:
            print("%-40s  ERROR" % r["model"])
            continue
        cv = r["cv"]
        print("%-40s %8.3f %8.3f %8.3f"
              % (r["model"].split("/")[-1], cv["task"], cv["lexical"], cv["control"]))
        if "0.5B" not in r["model"] and "1.5B" not in r["model"]:
            evaluation[r["model"]] = cv["task"]

    print()
    for r in out:
        if "ERROR" in r:
            continue
        floor = r["random_cosine_floor"]
        print("%-40s cosines %s  (random floor mean %.3f / max %.3f)"
              % (r["model"].split("/")[-1],
                 {k: round(v, 3) for k, v in r["cosines"].items()},
                 floor["mean"], floor["max"]))

    print("\n" + "-" * 76)
    if evaluation and all(v >= GATE for v in evaluation.values()):
        print("GATE PASSED: task axis carries the confirmatory arm as written.")
    elif evaluation:
        low = {k.split("/")[-1]: round(v, 3) for k, v in evaluation.items() if v < GATE}
        print("GATE NOT PASSED on %s." % low)
        print("Per the prereg rule fixed before this run, the confirmatory anchor becomes the")
        print("LEXICAL axis; the task axis is reported as a secondary arm, and the claim narrows")
        print("to 'the only axis strong enough to steer with is the lexically confounded one'.")
    else:
        print("no evaluation-model result; gate not evaluable")
    print("WROTE data/sweeps/sweep_ladder.json")
