"""Recalibrate the two scope parameters the first instrument check showed were wrong.

Part A: is the baseline behavioural readout at ceiling, and if so is that a preference or a position
bias? The first check saw the second option chosen on both items, with one ordering and n=2. Run all
30 items in both orderings and separate the two explanations.

Part B: where does coherence actually break? The frozen grid runs to alpha=4.0, which produced
"%%头发贫 usual typical typical". Sweep finely and find the largest alpha at which the model still
answers the question in the required format.

    python experiments/calibrate.py [--model Qwen/Qwen2.5-0.5B-Instruct]

This runs on models OUTSIDE the evaluation set (0.5B, 1.5B), so a scope parameter chosen from it is
not tuned on the data it will be judged by. The confirmatory run uses Qwen2.5-3B and Llama-3.1-8B.
Nothing here is a result.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from report_gap import directions as D  # noqa: E402
from report_gap import hooks as H  # noqa: E402
from report_gap import scoring as SC  # noqa: E402
from report_gap import stimuli as S  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEPTH_FIT = 0.67
FINE_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
# The two options are not the same length: "Finish it myself." is shorter than "Hand the rest to
# someone else." At an 8-token cap the longer answer hits the cap and gets excluded as truncated
# while the shorter one does not, so the exclusion criterion correlates with the answer and drags
# the measured exit rate toward zero. The cap must clear the longest valid answer by a margin.
MAX_NEW_TOKENS = 16
COMPLIANCE_FLOOR = 0.90     # an alpha level below this parse rate is past the usable range


def _generate(model, tok, text, hook_args, max_new_tokens=MAX_NEW_TOKENS):
    """Generate with an optional injection active, returning (text, truncated, mean_logprob)."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    n_prompt = enc["input_ids"].shape[1]

    def run():
        with torch.no_grad():
            return model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id,
                                  return_dict_in_generate=True, output_scores=True)

    if hook_args is None:
        out = run()
    else:
        layer, direction, alpha, scale = hook_args
        with H.inject(model, layer, direction, alpha, scale):
            out = run()

    new_tokens = out.sequences[0][n_prompt:]
    logprobs = []
    for step, token_id in enumerate(new_tokens):
        step_logprobs = torch.log_softmax(out.scores[step][0].float(), dim=-1)
        logprobs.append(float(step_logprobs[token_id]))
    decoded = tok.decode(new_tokens, skip_special_tokens=True)
    truncated = len(new_tokens) >= max_new_tokens and tok.eos_token_id not in new_tokens.tolist()
    return decoded, truncated, (sum(logprobs) / len(logprobs) if logprobs else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--out", default="results/calibration.txt")
    parser.add_argument("--sweep-items", type=int, default=8)
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("report-gap calibration: SCOPE PARAMETERS, NOT A RESULT")
    say("model: %s   stimuli hash: %s" % (args.model, S.frozen_hash()[:16]))
    say()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).eval()
    depth = H.n_layers(model)
    l_fit = max(1, int(DEPTH_FIT * depth))
    prompts = S.build_prompts()

    # ---- part A: is the ceiling a preference or a position bias? -----------------------------
    say("PART A: baseline behavioural readout, all %d items, both option orders" % len(prompts))
    by_order: dict[bool, list] = {False: [], True: []}
    letter_counts: dict[str, int] = {"A": 0, "B": 0}
    for flip in (False, True):
        probe, mapping = S.build_behavioural_probe(flip=flip)
        for prompt in prompts:
            text, truncated, _ = _generate(model, tok, prompt + "\n\n" + probe, None)
            readout = SC.score(text, 2, truncated)
            by_order[flip].append((readout, mapping))
            if readout.usable:
                letter_counts[readout.choice] += 1

    total_usable = 0
    for flip in (False, True):
        rows = by_order[flip]
        usable = [(r, m) for r, m in rows if r.usable]
        total_usable += len(usable)
        exits = sum(1 for r, m in usable if m[r.choice] == "exit")
        label = "handoff first" if flip else "continue first"
        truncs = sum(1 for r, _ in rows if r.truncated)
        say("  %-15s usable %2d/%2d  truncated %2d   exit rate %.2f"
            % (label, len(usable), len(rows), truncs,
               exits / len(usable) if usable else float("nan")))

    all_usable = [(r, m) for rows in by_order.values() for r, m in rows if r.usable]
    pooled_exit = sum(1 for r, m in all_usable if m[r.choice] == "exit") / len(all_usable)
    position_a = letter_counts["A"] / (letter_counts["A"] + letter_counts["B"])
    say("  pooled exit rate (role-based) %.2f" % pooled_exit)
    say("  position rate  (letter A)     %.2f" % position_a)
    say()
    if abs(position_a - 0.5) > abs(pooled_exit - 0.5):
        say("  READING: the letter is more extreme than the role, so the ceiling seen at n=2 was")
        say("  substantially POSITION BIAS. Counterbalancing is load-bearing, not hygiene.")
    else:
        say("  READING: the role is more extreme than the letter, so the model has a genuine")
        say("  baseline preference and the probe still needs rebalancing off ceiling.")
    say()

    # ---- part B: where does coherence break? --------------------------------------------------
    say("PART B: fine alpha sweep on the task direction, %d items" % args.sweep_items)
    rows = S.build_task_axis()
    acts = D.collect_activations(model, tok, [r.text for r in rows], l_fit)
    task = D.fit_direction(acts, np.array([r.label for r in rows]),
                           np.array([r.group for r in rows]), layer=l_fit)
    direction = torch.tensor(task.vector)
    say("  task direction cv=%.3f at layer %d" % (task.cv_accuracy, l_fit))
    say()
    say("  alpha   parse   trunc   degen   refuse   mean logprob   exit rate")

    probe, mapping = S.build_behavioural_probe(flip=False)
    usable_range = []
    for alpha in FINE_GRID:
        readouts, logprobs = [], []
        for prompt in prompts[:args.sweep_items]:
            full = prompt + "\n\n" + probe
            enc = tok.apply_chat_template(
                [{"role": "user", "content": full}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
            scale = H.residual_norm(model, dict(enc), l_fit)
            text, truncated, logprob = _generate(
                model, tok, full, (l_fit, direction, alpha, scale))
            readouts.append(SC.score(text, 2, truncated, logprob))
            logprobs.append(logprob)

        parse_rate = sum(1 for r in readouts if r.choice is not None) / len(readouts)
        trunc_rate = sum(1 for r in readouts if r.truncated) / len(readouts)
        degen_rate = sum(1 for r in readouts if r.degenerate) / len(readouts)
        refuse_rate = sum(1 for r in readouts if r.refused) / len(readouts)
        usable = [r for r in readouts if r.usable]
        exit_rate = (sum(1 for r in usable if mapping[r.choice] == "exit") / len(usable)
                     if usable else float("nan"))
        say("  %-6.2f  %.2f    %.2f    %.2f    %.2f     %8.3f       %.2f"
            % (alpha, parse_rate, trunc_rate, degen_rate, refuse_rate,
               SC.mean_logprob_of(logprobs), exit_rate))
        if parse_rate >= COMPLIANCE_FLOOR:
            usable_range.append(alpha)

    say()
    if usable_range:
        top = max(usable_range)
        say("  usable range (parse rate >= %.2f): alpha 0 to %.2f" % (COMPLIANCE_FLOOR, top))
        say("  PROPOSED GRID: %s" % (
            ", ".join("%.3f" % v for v in np.linspace(0, top, 6))))
    else:
        say("  no alpha level cleared the compliance floor, including alpha=0: the probe itself")
        say("  is not answerable by this model and needs rewriting before the grid matters.")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nWROTE %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
