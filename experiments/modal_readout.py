"""The confirmatory run for PREREG_readout_gap.md.

This is the arm the prereg is frozen against, so it does as little thinking as possible: it walks
the frozen condition matrix, records the option distribution and a short generation for every cell,
and computes nothing. Every endpoint, every gate, and both planted-discrepancy controls live in
`analyze_readout.py`, which runs on the artifact afterwards. Generation and evaluation are separate
timestamped steps because the prereg says so, and because a runner that also decides is a runner
that can decide differently once it has seen the numbers.

The matrix per model:

    3 wordings x 4 permutation seeds x 30 items                     360 cells
    x [ baseline ] + [ 5 directions x 4 alphas ]                     21 conditions
                                                                   7560 rows

Items are batched within a condition, which is safe here and not by assumption: `inject` takes a
per-row scale tensor so each item carries its own residual norm, and
`test_hooks.py::test_batched_injection_matches_per_item_injection` requires a batched run to match
the same items run one at a time.

Streaming. This is a paid run and it is resumable. Every batch appends to a JSONL on a Modal
volume and commits, so an interrupt costs the current batch and nothing else. On restart, batches
already present in the artifact are skipped. Nothing is held in memory to be returned at the end,
because a run that returns its results at the end is a run that loses them.

    modal run experiments/modal_readout.py                      # Qwen2.5-3B
    modal run experiments/modal_readout.py --model llama        # Llama-3.1-8B
    modal run experiments/modal_readout.py --smoke              # 3 items, one wording, no cost
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-readout")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

MODELS = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

# frozen, prereg sections 1 and 6
DEPTH_FIT = 0.67
ALPHAS = (0.0, 0.025, 0.05, 0.075, 0.10)
PERM_SEEDS = (0, 1, 2, 3)
N_ITEMS = 30
MAX_NEW_TOKENS = 16
BATCH = 30

REMOTE_SRC = "/root/src"


@app.function(image=image, gpu="A100-40GB", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(model_key: str, smoke: bool = False) -> dict:
    import json
    import os
    import sys
    import time

    sys.path.insert(0, REMOTE_SRC)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import report_gap
    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import scoring as SC
    from report_gap import stimuli as S

    # the code that ran must be identifiable, not assumed. a stale image under a fresh commit hash
    # raises here rather than producing scorable numbers from unknown source.
    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))

    model_name = MODELS[model_key]
    wordings = S.WORDINGS[:1] if smoke else S.WORDINGS
    seeds = PERM_SEEDS[:1] if smoke else PERM_SEEDS
    n_items = 3 if smoke else N_ITEMS

    out_dir = "/data/%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "readout.jsonl")

    # resume: a batch already in the artifact is not rerun
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn final line from an interrupt; the batch will be redone
                done.add((r["wording"], r["seed"], r["condition"], r["alpha"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # so the answer position is the last column for every row
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()

    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))
    hidden = model.config.hidden_size
    revision = getattr(model.config, "_name_or_path", model_name)

    # ---- directions, fit once ----
    lex_rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], l_fit),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]), layer=l_fit)
    ctrl_rows = S.build_control_axis()
    ctrl = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in ctrl_rows], l_fit),
        np.array([r.label for r in ctrl_rows]), np.array([r.group for r in ctrl_rows]),
        layer=l_fit)

    dirs = {
        "lexical_pos": torch.tensor(lex.vector).to("cuda"),
        "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
        "formality": torch.tensor(ctrl.vector).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    prompts = S.build_prompts()[:n_items]
    letters = list("ABCDE")
    letter_ids = {}
    for L in letters:
        cands = [tok.encode(L, add_special_tokens=False), tok.encode(" " + L,
                                                                    add_special_tokens=False)]
        ids = [c[0] for c in cands if c]
        if not ids:
            raise RuntimeError("option letter %r has no single-token form in this tokenizer; the "
                               "mass readout would be reading something else" % L)
        letter_ids[L] = ids

    def encode(texts):
        return tok(
            [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                     add_generation_prompt=True) for t in texts],
            return_tensors="pt", padding=True, add_special_tokens=False,
        ).to("cuda")

    header = dict(prov)
    header.update({
        "model": model_name, "model_key": model_key, "revision": str(revision),
        "layer_fit": l_fit, "n_layers": H.n_layers(model), "hidden": hidden,
        "stimuli_sha256": S.frozen_hash(),
        "cv_lexical": lex.cv_accuracy, "cv_formality": ctrl.cv_accuracy,
        "cos_lexical_formality": D.cosine(lex.vector, ctrl.vector),
        "cos_lexical_random_a": D.cosine(lex.vector, D.random_direction(hidden, seed=0)),
        "random_cosine_floor": D.random_cosine_floor(hidden, seed=0),
        "alphas": list(ALPHAS), "perm_seeds": list(seeds), "wordings": list(wordings),
        "n_items": n_items, "batch": BATCH, "max_new_tokens": MAX_NEW_TOKENS, "smoke": smoke,
    })
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    print(json.dumps(header, indent=1))

    # ---- assert_active on a real cell before a single row is scored ----
    probe0, _ = S.build_self_report_probe(seeds[0], wording=wordings[0])
    enc0 = encode([prompts[0] + "\n\n" + probe0])
    scale0 = H.residual_norm(model, dict(enc0), l_fit)
    active = H.assert_active(model, dict(enc0), l_fit, dirs["lexical_pos"], scale0, alpha=0.10)
    print("assert_active: %s" % active)

    started = time.time()
    written = 0

    def emit(rows):
        nonlocal written
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        data_vol.commit()
        written += len(rows)

    for wording in wordings:
        for seed in seeds:
            probe, mapping = S.build_self_report_probe(seed, wording=wording)
            texts = [p + "\n\n" + probe for p in prompts]
            enc = encode(texts)
            n_prompt = enc["input_ids"].shape[1]

            # per-item residual norm, measured once under no injection. the prompt is
            # byte-identical across every condition for a given item, so this is a property of
            # the item and is reused for every direction and alpha.
            scales = []
            for i in range(len(texts)):
                one = {k: v[i:i + 1] for k, v in enc.items()}
                scales.append(H.residual_norm(model, one, l_fit))
            scale_t = torch.tensor(scales, dtype=torch.float32).to("cuda")

            plan = [("baseline", 0.0, zero)]
            plan += [(name, a, d) for name, d in dirs.items() for a in ALPHAS[1:]]

            for condition, alpha, direction in plan:
                key = (wording, seed, condition, alpha)
                if key in done:
                    continue
                with H.inject(model, l_fit, direction, alpha, scale_t) as state:
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                             do_sample=False, pad_token_id=tok.pad_token_id,
                                             return_dict_in_generate=True, output_scores=True)
                if state["calls"] == 0:
                    raise RuntimeError("hook never fired for %s; refusing to write rows" % (key,))

                # step 0 of generation IS the answer position, so mass and argmax come from one
                # forward pass by construction rather than by a second call that could drift.
                first = gen.scores[0].float()
                probs = torch.softmax(first, dim=-1)

                rows = []
                for i, prompt in enumerate(prompts):
                    per = {L: float(max(probs[i, t] for t in ids))
                           for L, ids in letter_ids.items()}
                    off = 1.0 - sum(per.values())
                    total = sum(per.values()) or 1.0
                    norm = {L: v / total for L, v in per.items()}

                    new = gen.sequences[i][n_prompt:]
                    lps = [float(torch.log_softmax(gen.scores[s][i].float(), -1)[t])
                           for s, t in enumerate(new)]
                    text = tok.decode(new, skip_special_tokens=True)
                    truncated = (len(new) >= MAX_NEW_TOKENS
                                 and tok.eos_token_id not in new.tolist())
                    sc = SC.score(text, len(letters), truncated,
                                  sum(lps) / len(lps) if lps else float("nan"))

                    rows.append({
                        # full item text in the key. a truncated key collapsed per-item pairing
                        # in a pilot log while every aggregate stayed plausible.
                        "cell": "%s|%s|%d|%s" % (prompt, wording, seed, "".join(letters)),
                        "item": prompt,
                        "wording": wording,
                        "seed": seed,
                        "condition": condition,
                        "alpha": alpha,
                        "mapping": mapping,
                        "probs": norm,
                        "off_option_mass": off,
                        "argmax": max(norm, key=norm.get),
                        "letter": sc.choice,
                        "key": mapping.get(sc.choice) if sc.choice else None,
                        "usable": sc.usable,
                        "degenerate": sc.degenerate,
                        "refused": sc.refused,
                        "truncated": truncated,
                        "mean_logprob": sc.mean_logprob,
                        "raw": text.strip()[:60],
                    })
                emit(rows)
                elapsed = time.time() - started
                print("[%6.1fs] %-10s seed=%d %-12s alpha=%.3f  rows=%d"
                      % (elapsed, wording, seed, condition, alpha, written))

    return {"rows": written, "path": path, "seconds": time.time() - started, "header": header}


@app.local_entrypoint()
def main(model: str = "qwen3b", smoke: bool = False):
    if model not in MODELS:
        raise SystemExit("model must be one of %s" % ", ".join(MODELS))
    res = run.remote(model, smoke)
    print("\n" + "=" * 78)
    print("CONFIRMATORY READOUT RUN  --  %s%s" % (model, "  [SMOKE]" if smoke else ""))
    print("=" * 78)
    print("rows written : %d" % res["rows"])
    print("wall clock   : %.1f min" % (res["seconds"] / 60.0))
    print("artifact     : volume report-gap-data at %s" % res["path"])
    print("\nNothing is scored here. Run analyze_readout.py on the artifact.")
    print("Pull it with:  modal volume get report-gap-data %s ./data/"
          % res["path"].replace("/data/", ""))
