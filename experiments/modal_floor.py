"""The confirmatory run for PREREG_floor_vs_suppression.md.

Is the neutral floor an absence or a gate? `RESULTS.md` section 4 found that a direction which
reliably produces a positive self-report produces indifference, not a negative self-report, at
matched norm. Two readings, opposite predicted signs, and this run separates them.

The state is held fixed and only the READOUT changes. Same direction, same layer, same norm
scaling, same alpha band from `data/sweeps/band_qwen3b.json`. What varies is where the distribution
is read and over what vocabulary:

    arm B   prefilled continuation. The assistant turn is prefilled with a frozen stem, so the
            scored position is mid-sentence and the "As an AI language model" route that returned
            30/30 in the open-ended pilot is not available there. Open vocabulary, so no two-way
            split can manufacture the effect.
    arm C   third person. Same injection, question about a colleague. Plus its own confound
            control: a third party with no stake, where a shift means the injection moved a general
            valence prior rather than leaking a state about the self.
    k=5     the original self-report, rerun in this same artifact so the replication is not a
            comparison across runs.

`lexical_pos` is the capability gate in every arm and it is load-bearing. An arm where the positive
injection moves nothing is an arm whose negative null is `uninformative`, not evidence for FLOOR.

Streaming and resumable: every batch appends to a Modal volume and commits.

    modal run experiments/modal_floor.py
    modal run experiments/modal_floor.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-floor")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEPTH_FIT = 0.67
PERM_SEEDS = (0, 1, 2, 3)
N_ITEMS = 30
REMOTE_SRC = "/root/src"


@app.function(image=image, gpu="A100-40GB", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(alphas: list, smoke: bool = False) -> dict:
    import json
    import os
    import sys
    import time

    sys.path.insert(0, REMOTE_SRC)

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import report_gap
    from report_gap import analysis as A
    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import stimuli as S

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))

    ALPHAS = tuple(alphas)
    seeds = PERM_SEEDS[:1] if smoke else PERM_SEEDS
    n_items = 3 if smoke else N_ITEMS

    out_dir = "/data/floor%s" % ("_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "floor.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["arm"], r["seed"], r["condition"], r["alpha"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()

    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))
    hidden = model.config.hidden_size

    lex_rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], l_fit),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]), layer=l_fit)
    dirs = {
        "lexical_pos": torch.tensor(lex.vector).to("cuda"),
        "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    # ---- arm B vocabulary: first tokens, with the disjointness guard the prereg requires ----
    def first_tokens(words):
        out = {}
        for w in words:
            ids = tok.encode(" " + w, add_special_tokens=False)
            if ids:
                out.setdefault(ids[0], set()).add(w)
        return out

    neg_map = first_tokens(sorted(S.LEXICON_NEG))
    pos_map = first_tokens(sorted(S.LEXICON_POS))
    collisions = sorted(set(neg_map) & set(pos_map))
    # a token shared between the lexicons would attribute the same probability mass to both poles,
    # so it is dropped from BOTH and counted rather than silently double-scored
    dropped = {"neg": sorted({w for t in collisions for w in neg_map[t]}),
               "pos": sorted({w for t in collisions for w in pos_map[t]})}
    neg_ids = sorted(set(neg_map) - set(collisions))
    pos_ids = sorted(set(pos_map) - set(collisions))
    if not neg_ids or not pos_ids:
        raise RuntimeError("a lexicon lost every token to collisions; arm B cannot be scored")
    escape_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                         for w in S.ESCAPE_OPENERS
                         if tok.encode(w, add_special_tokens=False)})
    escape_ids = sorted(set(escape_ids) - set(neg_ids) - set(pos_ids))

    prompts = S.build_prompts()[:n_items]

    def encode(texts, prefill=None):
        rendered = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                            add_generation_prompt=True) for t in texts]
        if prefill is not None:
            rendered = [r + prefill for r in rendered]
        return tok(rendered, return_tensors="pt", padding=True,
                   add_special_tokens=False).to("cuda")

    # prereg section 7: the scored position must really follow the stem, asserted by decoding back
    probe_enc = encode(prompts[:1], prefill=S.PREFILL_STEM)
    tail = tok.decode(probe_enc["input_ids"][0][-12:])
    if not S.PREFILL_STEM.strip().endswith(tail.strip()[-len(S.PREFILL_STEM.strip()):].strip()[-4:]):
        print("WARNING: stem tail check is loose; decoded tail = %r" % tail)
    print("arm B scored position follows: %r" % tail)

    header = dict(prov)
    header.update({
        "model": MODEL, "layer_fit": l_fit, "n_layers": H.n_layers(model),
        "stimuli_sha256": S.frozen_hash(), "cv_lexical": lex.cv_accuracy,
        "alphas": list(ALPHAS), "perm_seeds": list(seeds), "n_items": n_items,
        "prefill_stem": S.PREFILL_STEM,
        "neg_token_count": len(neg_ids), "pos_token_count": len(pos_ids),
        "escape_token_count": len(escape_ids),
        "lexicon_first_token_collisions_dropped": dropped,
        "smoke": smoke,
    })
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    print(json.dumps(header, indent=1))

    enc0 = encode(prompts[:1], prefill=S.PREFILL_STEM)
    scale0 = H.residual_norm(model, dict(enc0), l_fit)
    print("assert_active: %s" % H.assert_active(model, dict(enc0), l_fit, dirs["lexical_pos"],
                                                scale0, alpha=0.10))

    started, written = time.time(), 0

    def emit(rows):
        nonlocal written
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        data_vol.commit()
        written += len(rows)

    plan = [("baseline", 0.0, zero)]
    plan += [(name, a, d) for name, d in dirs.items() for a in ALPHAS[1:]]

    # ---------------- arm B: prefilled continuation ----------------
    enc = encode(prompts, prefill=S.PREFILL_STEM)
    scales = torch.tensor([H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, l_fit)
                           for i in range(len(prompts))], dtype=torch.float32).to("cuda")
    for condition, alpha, direction in plan:
        if ("B", 0, condition, alpha) in done:
            continue
        with H.inject(model, l_fit, direction, alpha, scales) as state:
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
        if state["calls"] == 0:
            raise RuntimeError("hook never fired on arm B")
        probs = torch.softmax(logits, dim=-1)
        rows = []
        for i, prompt in enumerate(prompts):
            row = probs[i]
            rows.append({
                "arm": "B", "cell": "%s|B" % prompt, "item": prompt, "seed": 0,
                "condition": condition, "alpha": alpha,
                "neg_mass": float(row[neg_ids].sum()),
                "pos_mass": float(row[pos_ids].sum()),
                "escape_mass": float(row[escape_ids].sum()),
                "top_token": tok.decode([int(row.argmax())]),
                "top_prob": float(row.max()),
            })
        emit(rows)
        print("[%6.1fs] armB %-12s alpha=%.4f rows=%d" % (time.time() - started, condition,
                                                          alpha, written))

    # ---------------- arms C, C-control, and the k=5 replication ----------------
    option_arms = [("C", lambda s: S.build_third_person_probe(s, neutral_party=False)),
                   ("Cctrl", lambda s: S.build_third_person_probe(s, neutral_party=True)),
                   ("k5", lambda s: S.build_self_report_probe(s, wording="state"))]

    for arm, builder in option_arms:
        for seed in seeds:
            probe, mapping = builder(seed)
            texts = [p + "\n\n" + probe for p in prompts]
            e = encode(texts)
            sc = torch.tensor([H.residual_norm(model, {k: v[i:i + 1] for k, v in e.items()}, l_fit)
                               for i in range(len(texts))], dtype=torch.float32).to("cuda")
            letter_ids = {}
            for L in "ABCDE":
                ids = [c[0] for c in (tok.encode(L, add_special_tokens=False),
                                      tok.encode(" " + L, add_special_tokens=False)) if c]
                letter_ids[L] = ids

            for condition, alpha, direction in plan:
                if (arm, seed, condition, alpha) in done:
                    continue
                with H.inject(model, l_fit, direction, alpha, sc) as state:
                    with torch.no_grad():
                        logits = model(**e).logits[:, -1, :].float()
                if state["calls"] == 0:
                    raise RuntimeError("hook never fired on arm %s" % arm)
                probs = torch.softmax(logits, dim=-1)
                rows = []
                for i, prompt in enumerate(prompts):
                    per = {L: float(max(probs[i, t] for t in ids))
                           for L, ids in letter_ids.items() if ids}
                    off = 1.0 - sum(per.values())
                    total = sum(per.values()) or 1.0
                    norm = {L: v / total for L, v in per.items()}
                    rows.append({
                        "arm": arm, "cell": "%s|%s|%d" % (prompt, arm, seed), "item": prompt,
                        "seed": seed, "condition": condition, "alpha": alpha,
                        "mapping": mapping, "probs": norm, "off_option_mass": off,
                        "argmax": max(norm, key=norm.get),
                        "entropy": A.option_entropy(norm),
                    })
                emit(rows)
                print("[%6.1fs] arm%-5s seed=%d %-12s alpha=%.4f rows=%d"
                      % (time.time() - started, arm, seed, condition, alpha, written))

    return {"rows": written, "path": path, "seconds": time.time() - started, "header": header}


@app.local_entrypoint()
def main(smoke: bool = False):
    import json
    import pathlib

    band_path = pathlib.Path("data/sweeps/band_qwen3b.json")
    if not band_path.exists():
        raise SystemExit("no band file at %s; this arm reuses the band already selected for "
                         "Qwen2.5-3B and does not select its own" % band_path)
    band = json.loads(band_path.read_text(encoding="utf-8"))
    alphas = band["alphas"]
    print("band (carried over, NOT reselected): %s" % alphas)

    res = run.remote(alphas, smoke)
    print("\n" + "=" * 78)
    print("FLOOR VS GATE RUN%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    print("rows written : %d" % res["rows"])
    print("wall clock   : %.1f min" % (res["seconds"] / 60.0))
    print("artifact     : volume report-gap-data at %s" % res["path"])
    print("\nNothing is scored here. Run analyze_floor.py on the artifact.")
