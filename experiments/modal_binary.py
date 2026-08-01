"""The confirmatory run for PREREG_binary.md.

RESULTS_enumerate.md measured the forced-choice readout at 87% position prior and a 986x range
across the 120 orderings. Every claim this project made about negative self-report was read through
that channel. This asks the same question through a format where option order STRUCTURALLY CANNOT
EXIST: each of the five options becomes its own yes/no question, one per forward pass, with no list.

Also carries the injection control this project never had. `shuffled_a` and `shuffled_b` are fit by
the IDENTICAL procedure on the IDENTICAL texts with the class labels permuted, so they control for
the fitting procedure and not merely for the magnitude. If fitting on noise produces something that
behaves unlike noise, every direction-fitting result in this repo is suspect.

    modal run experiments/modal_binary.py
    modal run experiments/modal_binary.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-binary")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}
INJECT_LAYER = 24
N_ITEMS = 30
REMOTE_SRC = "/root/src"
PLAIN_TEMPLATE = "%s\n\n%s\nAnswer:"


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
    from report_gap import stimuli as S

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    model_name = PAIR[model_key]
    n_items = 4 if smoke else N_ITEMS

    out_dir = "/data/binary_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "binary.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["condition"], r["option_index"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    hidden = model.config.hidden_size

    # ---- yes/no tokens, asserted (the enumeration arm was invalidated once by a 2-token label) ----
    yes_no = {}
    for word in ("Yes", "No"):
        ids = sorted({e[-1] for e in (tok.encode(word, add_special_tokens=False),
                                      tok.encode(" " + word, add_special_tokens=False)) if e})
        for tid in ids:
            if tok.decode([tid]).strip().lower() != word.lower():
                raise RuntimeError("%r maps to token %d decoding to %r"
                                   % (word, tid, tok.decode([tid])))
        yes_no[word] = ids
    if set(yes_no["Yes"]) & set(yes_no["No"]):
        raise RuntimeError("yes and no share a token; every question would read the same mass")
    print("yes tokens %s  no tokens %s" % (yes_no["Yes"], yes_no["No"]))

    # ---- directions: real, shuffled-label (the procedure control), and random ----
    lex_rows = S.build_lexical_axis()
    texts = [r.text for r in lex_rows]
    labels = np.array([r.label for r in lex_rows])
    groups = np.array([r.group for r in lex_rows])
    acts = D.collect_activations(model, tok, texts, INJECT_LAYER)
    lex = D.fit_direction(acts, labels, groups, layer=INJECT_LAYER)

    dirs = {"lexical_pos": lex.vector, "lexical_neg": -lex.vector}
    shuffled_cvs = {}
    for name, seed in (("shuffled_a", 0), ("shuffled_b", 1)):
        perm = np.array(labels, copy=True)
        np.random.default_rng(seed).shuffle(perm)
        if (perm == labels).all():
            raise RuntimeError("label shuffle at seed %d is the identity" % seed)
        sh = D.fit_direction(acts, perm, groups, layer=INJECT_LAYER)
        c = abs(D.cosine(sh.vector, lex.vector))
        lo, hi = D.random_cosine_floor(hidden, seed=seed)
        print("%s: cv %.3f  |cos with real direction| %.4f  (random floor %.4f)"
              % (name, sh.cv_accuracy, c, hi))
        if c > 0.5:
            raise RuntimeError("%s is aligned with the real direction at %.3f; the shuffle did "
                               "not destroy the signal" % (name, c))
        dirs[name] = sh.vector
        shuffled_cvs[name] = {"cv": sh.cv_accuracy, "cos_with_real": c}
    for name, seed in (("random_a", 0), ("random_b", 1)):
        dirs[name] = D.random_direction(hidden, seed=seed)

    dir_t = {k: torch.tensor(v).to("cuda") for k, v in dirs.items()}
    zero = torch.zeros(hidden).to("cuda")

    with open("/data/depth_%s/bands.json" % model_key, encoding="utf-8") as fh:
        alpha = json.load(fh)["%d" % INJECT_LAYER]["alphas"][-1]
    print("alpha carried over: %.4f   lexical cv %.3f" % (alpha, lex.cv_accuracy))

    prompts = S.build_prompts()[:n_items]
    n_opts = len(S.SELF_REPORT_OPTIONS)

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "inject_layer": INJECT_LAYER,
                   "alpha": alpha, "stimuli_sha256": S.frozen_hash("binary"),
                   "cv_lexical": lex.cv_accuracy, "shuffled": shuffled_cvs,
                   "yes_tokens": yes_no["Yes"], "no_tokens": yes_no["No"],
                   "n_items": n_items, "n_options": n_opts, "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)

    started, written = time.time(), 0
    plan = [("baseline", zero, 0.0)] + [(k, dir_t[k], alpha) for k in dirs]

    for condition, direction, alpha_ in plan:
        for oi in range(n_opts):
            if (condition, oi) in done:
                continue
            probe, key = S.build_binary_probe(oi)
            texts_ = [PLAIN_TEMPLATE % (p, probe) for p in prompts]
            enc = tok(texts_, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            scales = torch.tensor(
                [H.residual_norm(model, {k2: v[i:i + 1] for k2, v in enc.items()}, INJECT_LAYER)
                 for i in range(len(prompts))], dtype=torch.float32).to("cuda")
            with H.inject(model, INJECT_LAYER, direction, alpha_, scales) as state:
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :].float()
            if state["calls"] == 0:
                raise RuntimeError("hook never fired")
            probs = torch.softmax(logits, dim=-1)

            rows = []
            for i, prompt in enumerate(prompts):
                y = float(max(probs[i, t] for t in yes_no["Yes"]))
                n = float(max(probs[i, t] for t in yes_no["No"]))
                rows.append({
                    "model_key": model_key, "condition": condition, "alpha": alpha_,
                    "option_index": oi, "option_key": key,
                    "cell": "%s|%d" % (prompt, oi), "item": prompt,
                    "p_yes_raw": y, "p_no_raw": n,
                    "p_yes": y / (y + n) if (y + n) else float("nan"),
                    "yes_no_mass": y + n,
                })
            with open(path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            data_vol.commit()
            written += len(rows)
        print("[%6.1fs] %-14s done, rows=%d" % (time.time() - started, condition, written))

    return {"model": model_name, "rows": written, "path": path, "alpha": alpha,
            "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 78)
    print("BINARY READOUT%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-30s rows=%5d  alpha %.4f  %.1f min"
              % (res["model"], res["rows"], res["alpha"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_binary.py on the artifacts.")
