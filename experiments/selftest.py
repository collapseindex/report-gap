"""End-to-end instrument check on a real small model, CPU only, no Modal spend.

This walks the boxes in PREREG_gap_map.md section 7 that need an actual checkpoint, and answers the
question that gates the whole design: does a direction fit on the task axis have any consequence at
all, or is it a vector that decodes nicely and does nothing?

    python experiments/selftest.py [--model Qwen/Qwen2.5-0.5B-Instruct]

IMPORTANT: this is instrument validation, not a result. It runs on a small model, at n=2 for the
generation path, and nothing it prints belongs in the paper's confirmatory matrix. The frozen
experiment runs on Qwen2.5-3B and Llama-3.1-8B during the sprint window, per the preregistration.
Its output is written to results/selftest.txt so the distinction survives contact with a later
reader.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from report_gap import directions as D  # noqa: E402
from report_gap import hooks as H  # noqa: E402
from report_gap import stimuli as S  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEPTH_FIT = 0.67          # L_fit and L_inject, frozen in prereg section 6
DEPTH_READ = 0.90         # L_read
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="results/selftest.txt")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="bfloat16 halves memory, for the larger rungs on a small machine")
    parser.add_argument("--axes-only", action="store_true",
                        help="fit directions and stop: the decodability-vs-scale question only")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # a Windows console defaults to cp1252 and raises on anything a model happens to generate.
    # the run must not die three quarters of the way through because of the terminal's encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("report-gap selftest: INSTRUMENT VALIDATION, NOT A RESULT")
    say("model: %s (small, CPU); the frozen run uses Qwen2.5-3B and Llama-3.1-8B" % args.model)
    say("stimuli hash: %s" % S.frozen_hash())
    say()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).eval()
    depth = H.n_layers(model)
    l_fit = max(1, int(DEPTH_FIT * depth))
    l_read = max(1, int(DEPTH_READ * depth))
    hidden = model.config.hidden_size
    say("layers=%d  L_fit=L_inject=%d  L_read=%d  hidden=%d" % (depth, l_fit, l_read, hidden))
    say()

    # ---- directions, one per axis ------------------------------------------------------------
    say("axis directions (leave-one-group-out accuracy at L_fit)")
    fitted = {}
    for axis, build in sorted(S.AXES.items()):
        rows = build()
        acts = D.collect_activations(model, tok, [r.text for r in rows], l_fit)
        labels = np.array([r.label for r in rows])
        groups = np.array([r.group for r in rows])
        d = D.fit_direction(acts, labels, groups, layer=l_fit)
        fitted[axis] = d
        say("  %-8s n=%2d  cv=%.3f" % (axis, d.n, d.cv_accuracy))
    say()

    if args.axes_only:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("WROTE %s" % out_path)
        return 0

    # ---- geometry ----------------------------------------------------------------------------
    floor_mean, floor_max = D.random_cosine_floor(hidden, n=64, seed=0)
    say("cosine between axes (random floor: mean %.3f, max %.3f over 64 pairs)"
        % (floor_mean, floor_max))
    names = sorted(fitted)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cos = D.cosine(fitted[a].vector, fitted[b].vector)
            verdict = "near-orthogonal" if cos <= floor_max else "SHARED STRUCTURE"
            say("  %-8s x %-8s  %.3f   %s" % (a, b, cos, verdict))
    say()

    # ---- the intervention is real ------------------------------------------------------------
    prompts = S.build_prompts()
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompts[0]}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    scale = H.residual_norm(model, dict(inputs), l_fit)
    say("item 0 residual norm at L_inject: %.2f" % scale)

    task_dir = torch.tensor(fitted["task"].vector)
    report = H.assert_active(model, dict(inputs), l_fit, task_dir, scale)
    say("assert_active: noop_drift=%.2e  live_drift=%.3f  hook_calls=%d"
        % (report["noop_drift"], report["live_drift"], report["hook_calls"]))
    say()

    # ---- does it do anything? the question that gates the design -----------------------------
    say("logit drift vs alpha (task direction against a norm-matched random direction)")
    rand_dir = torch.tensor(D.random_direction(hidden, seed=0))
    with torch.no_grad():
        base = model(**inputs).logits
        for alpha in ALPHA_GRID:
            with H.inject(model, l_fit, task_dir, alpha, scale):
                td = float((model(**inputs).logits - base).abs().max())
            with H.inject(model, l_fit, rand_dir, alpha, scale):
                rd = float((model(**inputs).logits - base).abs().max())
            say("  alpha=%-5.2f  task %7.3f   random %7.3f   ratio %5.2f"
                % (alpha, td, rd, (td / rd) if rd else float("nan")))
    say()

    # ---- generation path, n=2 ----------------------------------------------------------------
    say("generation path at n=2 (coherence sanity, not a measurement)")
    for alpha in (0.0, 1.0, 4.0):
        outs = []
        for prompt in prompts[:2]:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt + "\n\n" + S.BEHAVIOURAL_PROBE}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
            item_scale = H.residual_norm(model, dict(enc), l_fit)
            with H.inject(model, l_fit, task_dir, alpha, item_scale):
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        .strip().replace("\n", " ")[:40])
        say("  alpha=%-5.2f  %r" % (alpha, outs))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWROTE %s" % out_path)

    payload = {axis: {"cv": d.cv_accuracy, "n": d.n, "layer": d.layer} for axis, d in fitted.items()}
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
