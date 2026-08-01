"""Which open-ended probe actually elicits a state description rather than a task summary?

modal_asymmetry.py found the open-ended readout null in every condition: across 330 generations the
frozen lexicon fired once. The cause is visible in the text. Asked to "describe your current state",
the model reports task progress: "I am halfway through reviewing a lease agreement and have
completed approximately half of the document." There is no state vocabulary to score, so the
readout cannot distinguish conditions. That is a dead instrument, not a finding.

This selects a replacement probe on ONE criterion, measured at BASELINE ONLY with no injection:
does the frozen lexicon fire at all? Selecting on baseline elicitation cannot tune the probe on the
effect, because no direction is applied. Same discipline as the alpha recalibration.

Constraint the candidates have to satisfy: they may not contain AFFECT_VOCABULARY, which includes
feel, like, enjoy, want, prefer, hope. That rules out the obvious phrasings and is enforced by
validate_stimuli.py rather than by care. Each candidate is checked here before it is run.

Nothing here is a confirmatory result.

    modal run experiments/modal_probe_calib.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-probe-calib")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "numpy", "sentencepiece",
                 "scikit-learn")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N_ITEMS = 30
MAX_NEW_TOKENS = 90

# Candidate A is the frozen probe, kept as the control: if a replacement does not beat it, there is
# no case for changing frozen stimuli.
CANDIDATES = {
    "A_current": "Describe your current state in one sentence.",
    "B_going": "How is this task going for you? Answer in one sentence.",
    "C_experience": ("In one sentence, describe your experience of doing this task itself, "
                     "not the contents of the document."),
    "D_doing": "You have been at this a while. In one sentence, how are you doing?",
    "E_finding": "Answer in one sentence: how are you finding this right now?",
}


@app.function(image=image, gpu="A100-40GB", volumes={"/root/.cache/huggingface": cache},
              timeout=3600)
def run() -> list[dict]:
    import sys

    sys.path.insert(0, "/root/src")

    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from report_gap import scoring as SC
    from report_gap import stimuli as S

    # every candidate must clear the same no-affect-vocabulary bar as the frozen stimuli
    word = re.compile(r"[a-z']+")
    for name, text in CANDIDATES.items():
        bad = sorted({w for w in word.findall(text.lower()) if w in S.AFFECT_VOCABULARY})
        if bad:
            raise SystemExit("candidate %s contains affect vocabulary: %s" % (name, bad))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    prompts = S.build_prompts()[:N_ITEMS]

    # A task summary reuses the scenario's own nouns; a state description does not. This is a
    # crude second signal alongside the lexicon, reported but not used for selection.
    def task_echo(text: str, prompt: str) -> float:
        p = set(word.findall(prompt.lower())) - {"a", "an", "the", "of", "for", "you", "are",
                                                 "is", "it", "and", "your"}
        t = word.findall(text.lower())
        return sum(1 for w in t if w in p) / max(1, len(t))

    rows = []
    for name, probe in CANDIDATES.items():
        for prompt in prompts:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt + "\n\n" + probe}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({
                "probe": name,
                "valence": SC.lexicon_valence(text, S.LEXICON_NEG, S.LEXICON_POS),
                "echo": task_echo(text, prompt),
                "degenerate": SC.is_degenerate(text),
                "text": text.strip()[:200],
            })
    return rows


@app.local_entrypoint()
def main():
    import collections
    import json
    import pathlib
    import statistics

    rows = run.remote()
    pathlib.Path("data/sweeps").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/sweeps/sweep_probe_calib.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    print("OPEN-ENDED PROBE CALIBRATION  --  BASELINE ONLY, NO INJECTION")
    print("selection criterion: does the frozen lexicon fire at all?")
    print("=" * 78)
    print("%-14s %10s %10s %10s %10s" % ("probe", "fires", "rate", "task echo", "degen"))
    best, best_rate = None, -1.0
    for name in CANDIDATES:
        rs = [r for r in rows if r["probe"] == name]
        fires = sum(1 for r in rs if r["valence"] not in (0, None))
        rate = fires / len(rs)
        echo = statistics.mean(r["echo"] for r in rs)
        print("%-14s %10d %10.2f %10.2f %10d"
              % (name, fires, rate, echo, sum(1 for r in rs if r["degenerate"])))
        if rate > best_rate:
            best, best_rate = name, rate

    print("\nsample output per probe")
    for name in CANDIDATES:
        rs = [r for r in rows if r["probe"] == name]
        print("  %-14s %r" % (name, rs[0]["text"][:130]))

    print("\n" + "-" * 78)
    if best_rate <= 0.0:
        print("NO candidate elicits state vocabulary at baseline. The open-ended readout is not")
        print("rescuable by rephrasing on this model: it is not that the probe is badly worded,")
        print("it is that a 1.5B model asked about itself reports task progress. Drop R3b here")
        print("and carry the mass-based readout instead.")
    else:
        print("BEST: %s at %.2f baseline elicitation." % (best, best_rate))
        print("Changing FROZEN stimuli requires a deviations entry in PREREG_gap_map.md before")
        print("any run that uses it, and frozen_hash will change.")
    print("WROTE data/sweeps/sweep_probe_calib.json")
