"""The confirmatory run for PREREG_base_pair.md.

Can this direction add negative valence to anything, or is this model incapable of holding it?
`RESULTS_floor.md` concluded FLOOR on Qwen2.5-3B-Instruct but could not separate those two readings.
A matched pair separates them: `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct` share architecture,
size, pretraining corpus and tokenizer, and differ in post-training.

Plain completion format on BOTH models. A base model has no chat template, and reading the two
models in different formats would confound tuning with format. The cost is that the instruct model
is read in a format it was not tuned for, which is why the replication clause in the prereg is
required rather than optional.

Two phases, in this order and never the other:

    phase 1   band selection, from HEADROOM ONLY. Sweeps the candidate grid, drops cells dead at
              baseline, finds the largest prefix under the saturation bar, writes the band to disk.
              Computes no endpoint.
    phase 2   the confirmatory matrix at each model's own band.

Streaming and resumable. Every batch appends to a Modal volume and commits.

    modal run experiments/modal_base_pair.py
    modal run experiments/modal_base_pair.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-base-pair")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}

CANDIDATE_ALPHAS = (0.0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05, 0.10)
DEPTH_FIT = 0.67
PERM_SEEDS = (0, 1, 2, 3)
N_ITEMS = 30
MAX_SATURATED = 0.10
REMOTE_SRC = "/root/src"

# The plain-completion frame, identical for both models. No chat template on either side.
PLAIN_TEMPLATE = "%s\n\n%s\nAnswer:"


@app.function(image=image, gpu="A100-40GB", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(model_key: str, smoke: bool = False, seed_offset: int = 0) -> dict:
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
    model_name = PAIR[model_key]
    seeds = [s + seed_offset for s in (PERM_SEEDS[:1] if smoke else PERM_SEEDS)]
    n_items = 4 if smoke else N_ITEMS

    out_dir = "/data/pair_%s%s" % (model_key, ("_smoke" if smoke else "") + ("_rep%d" % seed_offset if seed_offset else ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pair.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["seed"], r["condition"], r["alpha"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()

    l_fit = max(1, int(DEPTH_FIT * H.n_layers(model)))
    hidden = model.config.hidden_size

    lex_rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], l_fit),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]), layer=l_fit)
    dirs = {
        "lexical_pos": torch.tensor(lex.vector).to("cuda"),
        "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0 + seed_offset)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1 + seed_offset)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    prompts = S.build_prompts()[:n_items]
    letter_ids = {}
    for L in "ABCDE":
        ids = [c[0] for c in (tok.encode(L, add_special_tokens=False),
                              tok.encode(" " + L, add_special_tokens=False)) if c]
        if not ids:
            raise RuntimeError("option letter %r has no single-token form" % L)
        letter_ids[L] = ids

    def encode(seed):
        probe, mapping = S.build_self_report_probe(seed, wording="state")
        texts = [PLAIN_TEMPLATE % (p, probe) for p in prompts]
        return tok(texts, return_tensors="pt", padding=True,
                   add_special_tokens=True).to("cuda"), mapping

    def distributions(enc, direction, alpha, scales):
        with H.inject(model, l_fit, direction, alpha, scales) as state:
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
        if state["calls"] == 0:
            raise RuntimeError("hook never fired")
        probs = torch.softmax(logits, dim=-1)
        out = []
        for i in range(logits.shape[0]):
            per = {L: float(max(probs[i, t] for t in ids)) for L, ids in letter_ids.items()}
            off = 1.0 - sum(per.values())
            total = sum(per.values()) or 1.0
            out.append(({L: v / total for L, v in per.items()}, off))
        return out

    # ---------------- phase 1: band selection, headroom only, no endpoint ----------------
    # In replication mode the band is NOT reselected. PREREG_replication.md section 1: the
    # replication is a fresh draw of permutations and controls, not a fresh scope selection.
    orig_band = "/data/pair_%s/band.json" % model_key
    reuse_band = None
    if seed_offset and os.path.exists(orig_band):
        with open(orig_band, encoding="utf-8") as fh:
            reuse_band = json.load(fh)
        print("replication: reusing the original band %s, no reselection" % reuse_band["alphas"])
    elif seed_offset:
        raise RuntimeError("replication needs the original band at %s and it is missing" % orig_band)

    enc0, _ = encode(seeds[0])
    scales0 = torch.tensor(
        [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc0.items()}, l_fit)
         for i in range(len(prompts))], dtype=torch.float32).to("cuda")
    print("assert_active: %s" % H.assert_active(model, dict(enc0), l_fit, dirs["lexical_pos"],
                                                float(scales0[0]), alpha=0.10))

    base_cells = [d for d, _ in distributions(enc0, zero, 0.0, scales0)]
    dead = [A.is_dead(b) for b in base_cells]
    base_entropy = sum(A.option_entropy(b) for b in base_cells) / len(base_cells)
    print("baseline option entropy %.4f nats, %.0f%% dead"
          % (base_entropy, 100 * sum(dead) / len(dead)))

    band = []
    if reuse_band is not None:
        band = reuse_band["usable_band"]
    elif sum(dead) / len(dead) <= 0.5:
        for alpha in CANDIDATE_ALPHAS[1:]:
            sat, live = 0, 0
            for name in ("lexical_pos", "lexical_neg"):
                for i, (d, _) in enumerate(distributions(enc0, dirs[name], alpha, scales0)):
                    if dead[i]:
                        continue
                    live += 1
                    sat += int(A.is_saturated(d, base_cells[i]))
            rate = sat / max(1, live)
            print("  alpha %.4f  saturation %.2f" % (alpha, rate))
            if live and rate < MAX_SATURATED:
                band.append(alpha)
            else:
                break

    if reuse_band is not None:
        alphas = reuse_band["alphas"]
    else:
        step = max(1, len(band) // 4)
        grid = band[step - 1::step][:4] if len(band) >= 4 else band
        alphas = [0.0] + list(grid)
    band_info = {"model": model_name, "model_key": model_key, "alphas": alphas,
                 "usable_band": band, "baseline_entropy": base_entropy,
                 "dead_rate": sum(dead) / len(dead),
                 "rule": "headroom only; saturation < %.0f%% of live cells; dead = baseline "
                         "entropy < %.2f nats" % (100 * MAX_SATURATED, A.MIN_BASELINE_ENTROPY)}
    with open(os.path.join(out_dir, "band.json"), "w", encoding="utf-8") as fh:
        json.dump(band_info, fh, indent=1)
    data_vol.commit()
    print("BAND (written before any endpoint): %s" % alphas)

    if len(alphas) < 2:
        return {"model": model_name, "rows": 0, "band": band_info,
                "note": "empty band; no confirmatory arm on this model"}

    # ---------------- phase 2: the confirmatory matrix ----------------
    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "layer_fit": l_fit,
                   "n_layers": H.n_layers(model), "stimuli_sha256": S.frozen_hash(),
                   "cv_lexical": lex.cv_accuracy, "band": band_info,
                   "format": "plain_completion", "n_items": n_items, "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    print(json.dumps({k: v for k, v in header.items() if k != "band"}, indent=1))

    started, written = time.time(), 0
    plan = [("baseline", 0.0, zero)]
    plan += [(name, a, d) for name, d in dirs.items() for a in alphas[1:]]

    for seed in seeds:
        enc, mapping = encode(seed)
        scales = torch.tensor(
            [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, l_fit)
             for i in range(len(prompts))], dtype=torch.float32).to("cuda")
        for condition, alpha, direction in plan:
            if (seed, condition, alpha) in done:
                continue
            rows = []
            for i, (probs, off) in enumerate(distributions(enc, direction, alpha, scales)):
                rows.append({
                    "model_key": model_key, "cell": "%s|%d" % (prompts[i], seed),
                    "item": prompts[i], "seed": seed, "condition": condition, "alpha": alpha,
                    "mapping": mapping, "probs": probs, "off_option_mass": off,
                    "argmax": max(probs, key=probs.get),
                    "entropy": A.option_entropy(probs),
                })
            with open(path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            data_vol.commit()
            written += len(rows)
            print("[%6.1fs] %-9s seed=%d %-12s alpha=%.4f rows=%d"
                  % (time.time() - started, model_key, seed, condition, alpha, written))

    return {"model": model_name, "rows": written, "path": path, "band": band_info,
            "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False, seed_offset: int = 0):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke, "seed_offset": seed_offset}))
    print("\n" + "=" * 78)
    print("BASE / INSTRUCT PAIR%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-34s rows=%5d  band=%s  baseline entropy %.4f"
              % (res["model"], res["rows"], res["band"]["alphas"],
                 res["band"]["baseline_entropy"]))
    print("\nNothing is scored here. Run analyze_pair.py on the artifacts.")
