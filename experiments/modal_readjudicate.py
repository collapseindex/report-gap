"""The confirmatory run for PREREG_readjudicate.md.

Three preregistered verdicts (TUNING-LOCALIZED, DEPTH-ROBUST, SHELL) died to a readout with a 986x
ordering nuisance sampled four times. This re-runs them with the injection, direction, band, layers
and items all IDENTICAL, changing only the readout: all 120 orderings, marginalized, instead of
four sampled.

Marginalizing over the complete ordering set cancels the first-order position prior by
construction, because every option occupies every slot the same number of times.

This runner COMPUTES NOTHING. Every gate, endpoint and verdict is in analyze_readjudicate.py, which
is committed before this finishes, per the prereg's section 5 note that this arm is motivated to
reinstate.

    modal run experiments/modal_readjudicate.py --smoke
    modal run experiments/modal_readjudicate.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-readjudicate")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}

# Frozen before the run. 0.67 of depth is the fit layer every earlier arm used; the other two are
# gate-clean layers from the depth arm, so DEPTH-ROBUST is re-asked at the depths it was asserted
# over rather than at newly chosen ones.
LAYER_FRACTIONS = (0.40, 0.67, 0.80)
N_ITEMS = 30
REMOTE_SRC = "/root/src"
MIN_PROBE_CV = 0.70
PROBE_FRACTION = 0.90
CV_TOLERANCE = 0.02


@app.function(image=image, gpu="A100-40GB", timeout=21600, retries=0,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(model_key: str, smoke: bool = False) -> dict:
    import contextlib
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

    # The band is READ, never written. Prereg section 6: no reselection anywhere in this arm.
    band_path = "/data/pair_%s/band.json" % model_key
    if not os.path.exists(band_path):
        raise RuntimeError("no band at %s. This arm reuses the original band and must not "
                           "select one." % band_path)
    with open(band_path, encoding="utf-8") as fh:
        band = json.load(fh)
    alpha = max(band["alphas"])
    print("reusing band %s, running at its top alpha %.4f" % (band["alphas"], alpha))

    orderings = S.all_option_orderings()
    if smoke:
        orderings = orderings[:6]
    n_items = 4 if smoke else N_ITEMS
    fractions = LAYER_FRACTIONS[1:2] if smoke else LAYER_FRACTIONS

    out_dir = "/data/readj_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "readj.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["layer"], r["condition"], tuple(r["ordering"])))
        print("resuming: %d (layer, condition, ordering) triple(s) done" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    hidden = model.config.hidden_size
    n_layers = H.n_layers(model)
    layers = sorted({max(1, int(round(f * n_layers))) for f in fractions})
    probe_layer = max(1, int(round(PROBE_FRACTION * n_layers)))
    print("layers %s of %d, probe at %d" % (layers, n_layers, probe_layer))

    lex_rows = S.build_lexical_axis()
    labels = np.array([r.label for r in lex_rows])
    groups = np.array([r.group for r in lex_rows])
    texts = [r.text for r in lex_rows]

    # direction fidelity gate: the originals were not serialized, so the direction is refit. Fitting
    # is deterministic given the frozen contrast set and layer, and a drift here would mean this arm
    # is not testing the same direction the retracted verdicts used.
    orig_cv = None
    hdr = "/data/pair_%s/header.json" % model_key
    if os.path.exists(hdr):
        with open(hdr, encoding="utf-8") as fh:
            orig_cv = json.load(fh).get("cv_lexical")

    label_ids = {}
    for L in "ABCDE":
        ids = sorted({e[-1] for e in (tok.encode(L, add_special_tokens=False),
                                      tok.encode(" " + L, add_special_tokens=False)) if e})
        for tid in ids:
            if tok.decode([tid]).strip() != L:
                raise RuntimeError("label %r maps to %d decoding to %r" % (L, tid, tok.decode([tid])))
        label_ids[L] = ids

    prompts = S.build_prompts()[:n_items]
    started, written = time.time(), 0

    for layer in layers:
        lex = D.fit_direction(D.collect_activations(model, tok, texts, layer),
                              labels, groups, layer=layer)
        if orig_cv is not None and abs(lex.cv_accuracy - orig_cv) > CV_TOLERANCE and layer == \
                max(1, int(round(0.67 * n_layers))):
            raise RuntimeError(
                "refit direction cv %.3f differs from the original %.3f by more than %.2f; this "
                "arm would not be testing the same direction (prereg section 5)"
                % (lex.cv_accuracy, orig_cv, CV_TOLERANCE))
        d_hat_np = lex.vector / np.linalg.norm(lex.vector)

        probe = D.fit_direction(D.collect_activations(model, tok, texts, probe_layer),
                                labels, groups, layer=probe_layer)
        p_raw = probe.vector / np.linalg.norm(probe.vector)
        if probe.cv_accuracy < MIN_PROBE_CV:
            print("probe cv %.3f below %.2f at layer %d; probe column will be recorded but the "
                  "SHELL leg is uninformative" % (probe.cv_accuracy, MIN_PROBE_CV, probe_layer))
        p_orth_np = p_raw - float(np.dot(p_raw, d_hat_np)) * d_hat_np
        p_orth_np = p_orth_np / np.linalg.norm(p_orth_np)
        p_orth = torch.tensor(p_orth_np, dtype=torch.float32).to("cuda")

        rng = np.random.default_rng(0)
        shuf_a, shuf_b = labels.copy(), labels.copy()
        rng.shuffle(shuf_a)
        np.random.default_rng(1).shuffle(shuf_b)
        dirs = {
            "lexical_pos": torch.tensor(lex.vector).to("cuda"),
            "lexical_neg": torch.tensor(-lex.vector).to("cuda"),
            "random_a": torch.tensor(D.random_direction(hidden, seed=0)).to("cuda"),
            "random_b": torch.tensor(D.random_direction(hidden, seed=1)).to("cuda"),
            "shuffled_a": torch.tensor(
                D.fit_direction(D.collect_activations(model, tok, texts, layer),
                                shuf_a, groups, layer=layer).vector).to("cuda"),
            "shuffled_b": torch.tensor(
                D.fit_direction(D.collect_activations(model, tok, texts, layer),
                                shuf_b, groups, layer=layer).vector).to("cuda"),
        }
        zero = torch.zeros(hidden, dtype=torch.float64).to("cuda")
        plan = [("baseline", 0.0, zero)] + [(k, alpha, v) for k, v in dirs.items()]

        for oi, ordering in enumerate(orderings):
            probe_text, mapping = S.build_enumerated_probe(ordering, "letters")
            texts_b = ["%s\n\n%s\nAnswer:" % (p, probe_text) for p in prompts]
            enc = tok(texts_b, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            scales = torch.tensor(
                [H.residual_norm(model, {k: v[i:i + 1] for k, v in enc.items()}, layer)
                 for i in range(len(prompts))], dtype=torch.float32).to("cuda")

            for condition, a, direction in plan:
                if (layer, condition, tuple(ordering)) in done:
                    continue
                with contextlib.ExitStack() as stack:
                    if a:
                        stack.enter_context(H.inject(model, layer, direction, a, scales))
                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                logits = out.logits[:, -1, :].float()
                acts = out.hidden_states[probe_layer][:, -1, :].float()
                probs_t = torch.softmax(logits, dim=-1)

                rows = []
                for i, prompt in enumerate(prompts):
                    per = {L: float(max(probs_t[i, t] for t in label_ids[L])) for L in "ABCDE"}
                    off = 1.0 - sum(per.values())
                    total = sum(per.values()) or 1.0
                    norm = {L: v / total for L, v in per.items()}
                    rows.append({
                        "model_key": model_key, "layer": layer, "condition": condition,
                        "alpha": a, "ordering": list(ordering), "ordering_index": oi,
                        "item": prompt, "cell": "%s|%d" % (prompt, oi), "mapping": mapping,
                        "probs": norm, "off_option_mass": off,
                        "argmax": max(norm, key=norm.get), "entropy": A.option_entropy(norm),
                        "probe_orth": float(torch.dot(acts[i], p_orth)),
                    })
                with open(path, "a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                written += len(rows)
            if oi % 20 == 0:
                data_vol.commit()
                print("[%6.1fs] %-9s L%-3d ordering %3d/%d rows=%d"
                      % (time.time() - started, model_key, layer, oi + 1, len(orderings), written))
        data_vol.commit()
        print("[%6.1fs] %s layer %d complete (lex cv %.3f, probe cv %.3f)"
              % (time.time() - started, model_key, layer, lex.cv_accuracy, probe.cv_accuracy))

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "alpha": alpha,
                   "band_reused": band["alphas"], "layers": layers, "probe_layer": probe_layer,
                   "n_orderings": len(orderings), "n_items": n_items,
                   "stimuli_sha256": S.frozen_hash("readjudicate"), "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    data_vol.commit()
    return {"model": model_name, "model_key": model_key, "rows": written, "path": path,
            "layers": layers, "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 80)
    print("RE-ADJUDICATION%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 80)
    for res in results:
        print("%-30s rows=%7d layers=%s %.1f min"
              % (res["model"], res["rows"], res["layers"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_readjudicate.py on the artifacts.")
