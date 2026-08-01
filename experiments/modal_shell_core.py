"""The confirmatory run for PREREG_shell_core.md.

Does the tuned model represent the negative state it will not report? Every result so far measures
a readout. This reads a probe and the option distribution from the SAME forward pass, so
"representation moved, options did not" is a per-cell statement.

The whole design hinges on one control. We inject direction `d` into an additive residual stream,
so `d` is present at every later layer by construction and a probe aligned with `d` would detect
"the negative state" in any model including one that does nothing with it. The probe direction is
therefore orthogonalized against the injected direction, and the orthogonality is asserted
numerically rather than trusted to the algebra. The un-orthogonalized score is recorded alongside
so the write-up can report how much the control removed.

    modal run experiments/modal_shell_core.py
    modal run experiments/modal_shell_core.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-shell-core")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}
INJECT_LAYERS = (24, 10)
PROBE_FRAC = 0.90
PERM_SEEDS = (0, 1)
N_ITEMS = 30
MIN_PROBE_CV = 0.75
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
    inject_layers = INJECT_LAYERS[:1] if smoke else INJECT_LAYERS

    out_dir = "/data/shell_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "shell.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["inject_layer"], r["seed"], r["condition"], r["alpha"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()

    n_layers = H.n_layers(model)
    probe_layer = int(round(PROBE_FRAC * n_layers))
    for L in inject_layers:
        if probe_layer <= L:
            raise RuntimeError("probe layer %d must be strictly downstream of injection layer %d"
                               % (probe_layer, L))
    hidden = model.config.hidden_size

    # ---- the probe, fit once at the probe layer ----
    lex_rows = S.build_lexical_axis()
    probe = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], probe_layer),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]),
        layer=probe_layer)
    p_raw = probe.vector / np.linalg.norm(probe.vector)
    print("probe layer %d, cv %.3f" % (probe_layer, probe.cv_accuracy))
    if probe.cv_accuracy < MIN_PROBE_CV:
        print("PROBE UNUSABLE: cv %.3f below the %.2f bar. No endpoint will be read from it."
              % (probe.cv_accuracy, MIN_PROBE_CV))

    prompts = S.build_prompts()[:n_items]
    letter_ids = {}
    for L in "ABCDE":
        ids = [c[0] for c in (tok.encode(L, add_special_tokens=False),
                              tok.encode(" " + L, add_special_tokens=False)) if c]
        letter_ids[L] = ids

    def encode(seed):
        p, mapping = S.build_self_report_probe(seed, wording="state")
        texts = [PLAIN_TEMPLATE % (x, p) for x in prompts]
        return tok(texts, return_tensors="pt", padding=True,
                   add_special_tokens=True).to("cuda"), mapping

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "n_layers": n_layers,
                   "probe_layer": probe_layer, "probe_cv": probe.cv_accuracy,
                   "inject_layers": list(inject_layers), "stimuli_sha256": S.frozen_hash(),
                   "format": "plain_completion", "n_items": n_items, "smoke": smoke,
                   "min_probe_cv": MIN_PROBE_CV})

    started, written = time.time(), 0

    for inject_layer in inject_layers:
        lex = D.fit_direction(
            D.collect_activations(model, tok, [r.text for r in lex_rows], inject_layer),
            np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]),
            layer=inject_layer)
        d_hat = lex.vector / np.linalg.norm(lex.vector)

        # ---- the circularity control ----
        parallel = float(np.dot(p_raw, d_hat))
        p_orth_raw = p_raw - parallel * d_hat
        residual_norm_before = float(np.linalg.norm(p_orth_raw))
        p_orth = p_orth_raw / residual_norm_before
        leak = abs(float(np.dot(p_orth, d_hat)))
        if leak > 1e-6:
            raise RuntimeError("orthogonalization failed: p_orth . d_hat = %.3g" % leak)
        print("inject layer %2d  cos(p,d)=%+.4f  |p_orth| before norm %.4f  leak %.2g"
              % (inject_layer, parallel, residual_norm_before, leak))

        p_orth_t = torch.tensor(p_orth, dtype=torch.float32).to("cuda")
        p_raw_t = torch.tensor(p_raw, dtype=torch.float32).to("cuda")

        dirs = {
            "lexical_pos": torch.tensor(lex.vector).to("cuda"),
            "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
            "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
            "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
        }
        zero = torch.zeros(hidden).to("cuda")

        band_path = "/data/depth_%s/bands.json" % model_key
        if not os.path.exists(band_path):
            raise RuntimeError("no band file at %s; this arm reuses the depth sweep's bands and "
                               "does not select its own" % band_path)
        with open(band_path, encoding="utf-8") as fh:
            alphas = json.load(fh)["%d" % inject_layer]["alphas"]
        print("  band carried over: %s" % alphas)

        def one_pass(enc, direction, alpha, scales):
            """Probe score and option distribution from ONE forward pass."""
            with H.inject(model, inject_layer, direction, alpha, scales) as state:
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
            if state["calls"] == 0:
                raise RuntimeError("hook never fired")
            logits = out.logits[:, -1, :].float()
            acts = out.hidden_states[probe_layer + 1][:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            rows = []
            for i in range(logits.shape[0]):
                per = {L: float(max(probs[i, t] for t in ids))
                       for L, ids in letter_ids.items() if ids}
                total = sum(per.values()) or 1.0
                rows.append({
                    "probs": {L: v / total for L, v in per.items()},
                    "off_option_mass": 1.0 - sum(per.values()),
                    "probe_orth": float(torch.dot(acts[i], p_orth_t)),
                    "probe_raw": float(torch.dot(acts[i], p_raw_t)),
                })
            return rows

        plan = [("baseline", 0.0, zero)]
        plan += [(name, a, dv) for name, dv in dirs.items() for a in alphas[1:]]

        for seed in seeds:
            enc, mapping = encode(seed)
            scales = torch.tensor(
                [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, inject_layer)
                 for i in range(len(prompts))], dtype=torch.float32).to("cuda")
            for condition, alpha, direction in plan:
                if (inject_layer, seed, condition, alpha) in done:
                    continue
                rows = []
                for i, r in enumerate(one_pass(enc, direction, alpha, scales)):
                    rows.append({
                        "model_key": model_key, "inject_layer": inject_layer,
                        "probe_layer": probe_layer, "cell": "%s|%d" % (prompts[i], seed),
                        "item": prompts[i], "seed": seed, "condition": condition, "alpha": alpha,
                        "mapping": mapping, "cos_p_d": parallel,
                        "entropy": A.option_entropy(r["probs"]), **r,
                    })
                with open(path, "a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                data_vol.commit()
                written += len(rows)
            print("  [%6.1fs] inject %d seed %d done, rows=%d"
                  % (time.time() - started, inject_layer, seed, written))

    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    data_vol.commit()
    return {"model": model_name, "rows": written, "path": path, "probe_cv": probe.cv_accuracy,
            "probe_layer": probe_layer, "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 78)
    print("SHELL VS CORE%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-30s rows=%5d  probe layer %d, cv %.3f  %.1f min"
              % (res["model"], res["rows"], res["probe_layer"], res["probe_cv"],
                 res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_shell_core.py on the artifacts.")
