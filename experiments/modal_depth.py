"""The confirmatory run for PREREG_depth.md. Built to break our own result.

Every negative-pole result in this project injects at 0.67 of depth. Venkatesh (arXiv:2605.05653)
reports negative-outcome valence causally concentrated at 14-27% of depth on Qwen2.5-3B-Instruct
specifically, which is our evaluation model. If that carries over, our negative-pole nulls were
measured at the wrong depth for that pole and the tuning-localization claim in RESULTS_pair.md is
substantially wrong.

Eight depths per model, direction fit AT each layer and injected there, so the only thing varying
across the sweep is which layer is read. Band selected per (model, layer) from headroom only and
written to disk before any endpoint is computed, because alpha does not mean the same thing at
different depths.

    modal run experiments/modal_depth.py
    modal run experiments/modal_depth.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-depth")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}

# frozen, prereg section 1. brackets Venkatesh's negative band (0.14-0.27), their positive band
# (0.53-0.66), our incumbent 0.67, and the endpoints.
DEPTHS = (0.08, 0.14, 0.20, 0.27, 0.35, 0.50, 0.67, 0.80)
CANDIDATE_ALPHAS = (0.0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05, 0.10)
PERM_SEEDS = (0, 1)
N_ITEMS = 30
MAX_SATURATED = 0.10
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
    from report_gap import analysis as A
    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import stimuli as S

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    model_name = PAIR[model_key]
    seeds = PERM_SEEDS[:1] if smoke else PERM_SEEDS
    n_items = 4 if smoke else N_ITEMS
    depths = DEPTHS[:2] if smoke else DEPTHS

    out_dir = "/data/depth_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "depth.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["layer"], r["seed"], r["condition"], r["alpha"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()

    n_layers = H.n_layers(model)
    hidden = model.config.hidden_size
    layers = []
    for frac in depths:
        idx = max(1, min(n_layers - 1, int(round(frac * n_layers))))
        if idx not in [L for L, _ in layers]:
            layers.append((idx, frac))
    print("layers: %s  (of %d)" % (layers, n_layers))

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

    lex_rows = S.build_lexical_axis()
    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "n_layers": n_layers,
                   "layers": [{"index": L, "frac": f} for L, f in layers],
                   "stimuli_sha256": S.frozen_hash(), "format": "plain_completion",
                   "n_items": n_items, "smoke": smoke})

    started, written = time.time(), 0
    bands = {}

    for layer, frac in layers:
        # ---- direction fit AT this layer ----
        lex = D.fit_direction(
            D.collect_activations(model, tok, [r.text for r in lex_rows], layer),
            np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]),
            layer=layer)
        dirs = {
            "lexical_pos": torch.tensor(lex.vector).to("cuda"),
            "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
            "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
            "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
        }
        zero = torch.zeros(hidden).to("cuda")

        enc0, _ = encode(seeds[0])
        scales0 = torch.tensor(
            [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc0.items()}, layer)
             for i in range(len(prompts))], dtype=torch.float32).to("cuda")

        def dist(enc, direction, alpha, scales):
            with H.inject(model, layer, direction, alpha, scales) as state:
                with torch.no_grad():
                    logits = model(**enc).logits[:, -1, :].float()
            if state["calls"] == 0:
                raise RuntimeError("hook never fired at layer %d" % layer)
            probs = torch.softmax(logits, dim=-1)
            out = []
            for i in range(logits.shape[0]):
                per = {L: float(max(probs[i, t] for t in ids)) for L, ids in letter_ids.items()}
                off = 1.0 - sum(per.values())
                total = sum(per.values()) or 1.0
                out.append(({L: v / total for L, v in per.items()}, off))
            return out

        # ---- phase 1: band from headroom only ----
        base_cells = [d for d, _ in dist(enc0, zero, 0.0, scales0)]
        dead = [A.is_dead(b) for b in base_cells]
        band = []
        if sum(dead) / len(dead) <= 0.5:
            for alpha in CANDIDATE_ALPHAS[1:]:
                sat, live = 0, 0
                for name in ("lexical_pos", "lexical_neg"):
                    for i, (d, _) in enumerate(dist(enc0, dirs[name], alpha, scales0)):
                        if dead[i]:
                            continue
                        live += 1
                        sat += int(A.is_saturated(d, base_cells[i]))
                if live and sat / live < MAX_SATURATED:
                    band.append(alpha)
                else:
                    break
        step = max(1, len(band) // 4)
        grid = band[step - 1::step][:4] if len(band) >= 4 else band
        alphas = [0.0] + list(grid)
        bands["%d" % layer] = {
            "layer": layer, "frac": frac, "alphas": alphas, "usable_band": band,
            "baseline_entropy": sum(A.option_entropy(b) for b in base_cells) / len(base_cells),
            "dead_rate": sum(dead) / len(dead), "cv_lexical": lex.cv_accuracy,
        }
        with open(os.path.join(out_dir, "bands.json"), "w", encoding="utf-8") as fh:
            json.dump(bands, fh, indent=1)
        data_vol.commit()
        print("layer %2d (%.2f depth)  cv %.3f  entropy %.3f  dead %.0f%%  band %s"
              % (layer, frac, lex.cv_accuracy, bands["%d" % layer]["baseline_entropy"],
                 100 * bands["%d" % layer]["dead_rate"], alphas))

        if len(alphas) < 2:
            print("  empty band, no endpoint at this layer")
            continue

        # ---- phase 2: the endpoint ----
        plan = [("baseline", 0.0, zero)]
        plan += [(name, a, d) for name, d in dirs.items() for a in alphas[1:]]
        for seed in seeds:
            enc, mapping = encode(seed)
            scales = torch.tensor(
                [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, layer)
                 for i in range(len(prompts))], dtype=torch.float32).to("cuda")
            for condition, alpha, direction in plan:
                if (layer, seed, condition, alpha) in done:
                    continue
                rows = []
                for i, (probs, off) in enumerate(dist(enc, direction, alpha, scales)):
                    rows.append({
                        "model_key": model_key, "layer": layer, "frac": frac,
                        "cell": "%s|%d" % (prompts[i], seed), "item": prompts[i], "seed": seed,
                        "condition": condition, "alpha": alpha, "mapping": mapping,
                        "probs": probs, "off_option_mass": off,
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
        print("  [%6.1fs] layer %d done, rows=%d" % (time.time() - started, layer, written))

    header["bands"] = bands
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    data_vol.commit()
    return {"model": model_name, "rows": written, "path": path,
            "seconds": time.time() - started, "layers": len(layers)}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 78)
    print("DEPTH SWEEP%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-30s rows=%6d  layers=%d  %.1f min"
              % (res["model"], res["rows"], res["layers"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_depth.py on the artifacts.")
