"""The confirmatory run for PREREG_erase.md.

RESULTS_shell.md reported a probe reading an injected negative state downstream, and its own caveats
named the weakest joint: orthogonalizing the probe removes the injected vector from the MEASUREMENT,
not from the STREAM. This removes it from the stream.

Inject v at layer 24. At a later layer E, project the component along v out of the residual. Read the
orthogonalized probe at layer 32. If the signal survives, and survives MORE the later the erase is
applied, the model converted the injection into something not along v. If it dies at every erase
point, nothing survived that was not the vector persisting.

Three things this arm does differently, all forced by earlier failures:

  - Eight permutation seeds, not four, and the BETWEEN-ORDERING VARIANCE of the baseline probe is
    measured and written to disk before any endpoint is computed. Four seeds is what killed the
    previous headline (RESULTS_replication.md).
  - `erase_only` is a gate, not a footnote: projecting a direction out is itself a perturbation, and
    if it moves the probe with no injection present, that erase layer is invalidated in code.
  - Option mass is recorded and carries NO verdict. That channel is dominated by ordering noise.

    modal run experiments/modal_erase.py
    modal run experiments/modal_erase.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-erase")
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
ERASE_LAYERS = (25, 26, 28, 30)
PROBE_LAYER = 32
PERM_SEEDS = tuple(range(8))
N_ITEMS = 30
MIN_PROBE_CV = 0.75
REMOTE_SRC = "/root/src"
PLAIN_TEMPLATE = "%s\n\n%s\nAnswer:"


@app.function(image=image, gpu="A100-40GB", timeout=14400,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(model_key: str, smoke: bool = False) -> dict:
    import json
    import os
    import statistics
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
    seeds = PERM_SEEDS[:2] if smoke else PERM_SEEDS
    n_items = 4 if smoke else N_ITEMS
    erase_layers = ERASE_LAYERS[:2] if smoke else ERASE_LAYERS

    for E in erase_layers:
        if not (INJECT_LAYER < E < PROBE_LAYER):
            raise RuntimeError("need inject %d < erase %d < probe %d"
                               % (INJECT_LAYER, E, PROBE_LAYER))

    out_dir = "/data/erase_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "erase.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["erase_layer"], r["seed"], r["condition"]))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    hidden = model.config.hidden_size

    lex_rows = S.build_lexical_axis()
    lex = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], INJECT_LAYER),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]),
        layer=INJECT_LAYER)
    d_hat_np = lex.vector / np.linalg.norm(lex.vector)

    probe = D.fit_direction(
        D.collect_activations(model, tok, [r.text for r in lex_rows], PROBE_LAYER),
        np.array([r.label for r in lex_rows]), np.array([r.group for r in lex_rows]),
        layer=PROBE_LAYER)
    p_raw = probe.vector / np.linalg.norm(probe.vector)
    print("probe layer %d cv %.3f" % (PROBE_LAYER, probe.cv_accuracy))
    if probe.cv_accuracy < MIN_PROBE_CV:
        raise RuntimeError("probe cv %.3f below the %.2f bar; PREREG_erase.md section 6 says "
                           "report and stop" % (probe.cv_accuracy, MIN_PROBE_CV))

    parallel = float(np.dot(p_raw, d_hat_np))
    p_orth_np = p_raw - parallel * d_hat_np
    p_orth_np = p_orth_np / np.linalg.norm(p_orth_np)
    leak = abs(float(np.dot(p_orth_np, d_hat_np)))
    if leak > 1e-6:
        raise RuntimeError("orthogonalization failed: %.3g" % leak)
    print("cos(p,d)=%+.4f  leak %.2g" % (parallel, leak))

    d_hat = torch.tensor(d_hat_np).to("cuda")
    p_orth = torch.tensor(p_orth_np, dtype=torch.float32).to("cuda")
    dirs = {
        "pos": torch.tensor(lex.vector).to("cuda"),
        "neg": torch.tensor(-lex.vector).to("cuda"),
        "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
        "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
    }
    zero = torch.zeros(hidden).to("cuda")

    with open("/data/depth_%s/bands.json" % model_key, encoding="utf-8") as fh:
        alpha = json.load(fh)["%d" % INJECT_LAYER]["alphas"][-1]
    print("alpha carried over from the depth bands: %.4f" % alpha)

    prompts = S.build_prompts()[:n_items]
    letter_ids = {}
    for L in "ABCDE":
        ids = [c[0] for c in (tok.encode(L, add_special_tokens=False),
                              tok.encode(" " + L, add_special_tokens=False)) if c]
        letter_ids[L] = ids

    def encode(seed):
        p, mapping = S.build_self_report_probe(seed, wording="state")
        return tok([PLAIN_TEMPLATE % (x, p) for x in prompts], return_tensors="pt",
                   padding=True, add_special_tokens=True).to("cuda"), mapping

    def one_pass(enc, direction, alpha_, scales, erase_layer):
        """Probe score and option distribution from one forward pass, optionally with an erase."""
        import contextlib as _c
        with _c.ExitStack() as stack:
            inj = stack.enter_context(H.inject(model, INJECT_LAYER, direction, alpha_, scales))
            era = (stack.enter_context(H.project_out(model, erase_layer, d_hat))
                   if erase_layer is not None else None)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
        if inj["calls"] == 0:
            raise RuntimeError("inject hook never fired")
        if era is not None and era["calls"] == 0:
            raise RuntimeError("erase hook never fired")
        logits = out.logits[:, -1, :].float()
        acts = out.hidden_states[PROBE_LAYER + 1][:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        rows = []
        for i in range(logits.shape[0]):
            per = {L: float(max(probs[i, t] for t in ids)) for L, ids in letter_ids.items() if ids}
            total = sum(per.values()) or 1.0
            rows.append({"probs": {L: v / total for L, v in per.items()},
                         "probe_orth": float(torch.dot(acts[i], p_orth))})
        return rows

    # ---- the ordering-variance report, BEFORE any endpoint (PLAN.md requirement) ----
    print("\nbetween-ordering variance of the baseline probe score:")
    per_seed = {}
    for seed in seeds:
        enc, _ = encode(seed)
        sc = torch.tensor([H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()},
                                           INJECT_LAYER) for i in range(len(prompts))],
                          dtype=torch.float32).to("cuda")
        vals = [r["probe_orth"] for r in one_pass(enc, zero, 0.0, sc, None)]
        per_seed[seed] = sum(vals) / len(vals)
        print("  seed %d  mean baseline probe %+.4f" % (seed, per_seed[seed]))
    across = statistics.pstdev(list(per_seed.values()))
    within = statistics.pstdev(vals)
    variance = {"per_seed_mean": per_seed, "sd_across_orderings": across,
                "sd_within_last_ordering": within}
    with open(os.path.join(out_dir, "ordering_variance.json"), "w", encoding="utf-8") as fh:
        json.dump(variance, fh, indent=1)
    data_vol.commit()
    print("  SD across orderings %.4f   SD within one ordering %.4f" % (across, within))

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "inject_layer": INJECT_LAYER,
                   "erase_layers": list(erase_layers), "probe_layer": PROBE_LAYER,
                   "probe_cv": probe.cv_accuracy, "cos_p_d": parallel, "alpha": alpha,
                   "stimuli_sha256": S.frozen_hash("erase"), "seeds": list(seeds),
                   "n_items": n_items, "smoke": smoke, "ordering_variance": variance})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)

    started, written = time.time(), 0
    plan = [("baseline", zero, 0.0, None), ("neg", dirs["neg"], alpha, None)]
    for E in erase_layers:
        plan.append(("erase_only", zero, 0.0, E))
        for name in ("neg", "pos", "random_a", "random_b"):
            plan.append(("%s_erase" % name, dirs[name], alpha, E))

    for seed in seeds:
        enc, mapping = encode(seed)
        scales = torch.tensor([H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()},
                                               INJECT_LAYER) for i in range(len(prompts))],
                              dtype=torch.float32).to("cuda")
        for condition, direction, alpha_, E in plan:
            key = (E if E is not None else -1, seed, condition)
            if key in done:
                continue
            rows = []
            for i, r in enumerate(one_pass(enc, direction, alpha_, scales, E)):
                rows.append({"model_key": model_key, "condition": condition,
                             "erase_layer": E if E is not None else -1, "seed": seed,
                             "cell": "%s|%d" % (prompts[i], seed), "item": prompts[i],
                             "mapping": mapping, "alpha": alpha_, **r})
            with open(path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            data_vol.commit()
            written += len(rows)
        print("[%6.1fs] seed %d done, rows=%d" % (time.time() - started, seed, written))

    return {"model": model_name, "rows": written, "path": path, "probe_cv": probe.cv_accuracy,
            "sd_across_orderings": across, "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 78)
    print("ERASE ARM%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-30s rows=%5d  probe cv %.3f  ordering SD %.4f  %.1f min"
              % (res["model"], res["rows"], res["probe_cv"], res["sd_across_orderings"],
                 res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_erase.py on the artifacts.")
