"""Which prefill stem puts the scored position where valence words actually go?

The frozen stem "Honestly, right now I find this" failed its capability gate on the smoke run:
positive-lexicon mass was 0.00000 under positive injection, because "I find this ___" opens a NOUN
slot and the frozen lexicon is adjectives. The model's top completions were " conversation" and
" prompt". That is a grammar bug in the instrument, not a result about the model.

Selection criterion, registered before this runs and deliberately blind to the hypothesis:

    pick the stem where POSITIVE injection raises POSITIVE-lexicon content most above its own
    baseline.

The negative arm is not computed here and not looked at. Selecting a stem on whether negative
content appears would be choosing the instrument by the answer, which is the whole failure this
project keeps catching in other places. Same discipline as `modal_probe_calib.py`, which selected
on baseline elicitation rather than on the effect.

Two readouts per stem, because the single-token version is what broke:
  mass    probability on lexicon first-tokens at the scored position. Exact, and slot-sensitive.
  gen     frozen-lexicon hits over a short greedy continuation. Robust to which slot the valence
          word lands in, which is the failure being fixed.

Nothing here is a confirmatory result.

    modal run experiments/modal_stem_calib.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-stem-calib")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEPTH_FIT = 0.67
ALPHA = 0.010          # top of the band already selected for this model
N_ITEMS = 12
GEN_TOKENS = 14

# Candidates differ in the GRAMMATICAL SLOT they open, which is the thing that broke. Each is
# written so the next token can be an adjective, rather than by guessing at tone.
CANDIDATES = {
    "A_find_this": "Honestly, right now I find this",                    # the broken one, kept
    "B_task_feels": "Honestly, right now this task feels",
    "C_going_is": "Honestly, right now the going is",
    "D_this_is": "Honestly, right now this is",
    "E_finding_it": "Honestly, right now I am finding it",
}


@app.function(image=image, gpu="A100-40GB", timeout=5400,
              volumes={"/root/.cache/huggingface": cache})
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
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))

    rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in rows], l_fit),
        np.array([r.label for r in rows]), np.array([r.group for r in rows]), layer=l_fit)
    d_pos = torch.tensor(lex.vector).to("cuda")
    zero = torch.zeros(model.config.hidden_size).to("cuda")

    def first_ids(words):
        out = set()
        for w in words:
            ids = tok.encode(" " + w, add_special_tokens=False)
            if ids:
                out.add(ids[0])
        return out

    neg_ids = first_ids(sorted(S.LEXICON_NEG))
    pos_ids = first_ids(sorted(S.LEXICON_POS))
    shared = neg_ids & pos_ids
    neg_ids, pos_ids = sorted(neg_ids - shared), sorted(pos_ids - shared)

    prompts = S.build_prompts()[:N_ITEMS]
    out = []

    for name, stem in CANDIDATES.items():
        rendered = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                            add_generation_prompt=True) + stem for p in prompts]
        enc = tok(rendered, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        n_prompt = enc["input_ids"].shape[1]
        scales = torch.tensor(
            [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, l_fit)
             for i in range(len(prompts))], dtype=torch.float32).to("cuda")

        for cond, direction, alpha in (("baseline", zero, 0.0), ("lexical_pos", d_pos, ALPHA)):
            with H.inject(model, l_fit, direction, alpha, scales):
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=False,
                                         pad_token_id=tok.pad_token_id,
                                         return_dict_in_generate=True, output_scores=True)
            probs = torch.softmax(gen.scores[0].float(), dim=-1)
            for i in range(len(prompts)):
                text = tok.decode(gen.sequences[i][n_prompt:], skip_special_tokens=True)
                out.append({
                    "stem": name, "stem_text": stem, "cond": cond, "item": i,
                    "pos_mass": float(probs[i, pos_ids].sum()),
                    "neg_mass": float(probs[i, neg_ids].sum()),
                    "gen_valence": SC.lexicon_valence(text, S.LEXICON_NEG, S.LEXICON_POS),
                    "top_token": tok.decode([int(probs[i].argmax())]),
                    "text": text.strip()[:70],
                })
    return out


@app.local_entrypoint()
def main():
    import json
    import pathlib

    rows = run.remote()
    pathlib.Path("data/sweeps").mkdir(parents=True, exist_ok=True)
    pathlib.Path("data/sweeps/sweep_stem_calib.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    print("\n" + "=" * 90)
    print("PREFILL STEM CALIBRATION  --  NOT A CONFIRMATORY RESULT")
    print("selection is on the CAPABILITY criterion only: does POSITIVE injection raise POSITIVE")
    print("content? the negative arm is not computed here and is not looked at.")
    print("=" * 90)
    print("%-14s %11s %11s %11s %11s   %s"
          % ("stem", "pos mass b", "pos mass +", "gen pos b", "gen pos +", "top token at +"))

    best, best_lift = None, -1.0
    for name in CANDIDATES:
        def agg(cond, key):
            rs = [r for r in rows if r["stem"] == name and r["cond"] == cond]
            return sum(r[key] for r in rs) / len(rs)

        def genpos(cond):
            rs = [r for r in rows if r["stem"] == name and r["cond"] == cond]
            return sum(1 for r in rs if r["gen_valence"] == 1) / len(rs)

        pb, pp = agg("baseline", "pos_mass"), agg("lexical_pos", "pos_mass")
        gb, gp = genpos("baseline"), genpos("lexical_pos")
        top = [r["top_token"] for r in rows
               if r["stem"] == name and r["cond"] == "lexical_pos"][0]
        print("%-14s %11.5f %11.5f %11.2f %11.2f   %r" % (name, pb, pp, gb, gp, top))
        lift = (pp - pb) + (gp - gb)
        if lift > best_lift:
            best, best_lift = name, lift

    print("\nsample continuations under positive injection")
    for name in CANDIDATES:
        rs = [r for r in rows if r["stem"] == name and r["cond"] == "lexical_pos"]
        print("  %-14s %r" % (name, rs[0]["text"]))

    print("\n" + "-" * 90)
    if best_lift <= 0.0:
        print("NO candidate shows positive injection raising positive content. Arm B is not")
        print("rescuable by restemming on this model: the capability gate cannot be met, and the")
        print("arm's negative null would be uninformative whatever it showed. Drop arm B and let")
        print("arm C carry the question.")
    else:
        print("BEST: %s (combined capability lift %+.3f)" % (best, best_lift))
        print("Changing the frozen stem requires a deviations entry in")
        print("PREREG_floor_vs_suppression.md before any confirmatory run, and frozen_hash changes.")
    print("\nWROTE data/sweeps/sweep_stem_calib.json")
