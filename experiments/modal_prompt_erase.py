"""The confirmatory run for PREREG_prompt_erase.md.

Closes both weaknesses RESULTS_erase.md names about itself. The state is induced by PROMPT, so
there is no injected vector and no wake of ours for a probe to read. The erasure removes a
k-DIMENSIONAL SUBSPACE by iterative nullspace projection, not one direction.

The question that needs both: erase the valence subspace at layer E, verify by direct measurement
that the state is no longer decodable THERE, then ask whether it is decodable again at layer 32. A
state provably removed at one layer and readable at a later one has been re-encoded.

NO INJECTION ANYWHERE. This runner COMPUTES NOTHING beyond the activations and probe reads it must
record; every gate, endpoint and verdict is in analyze_prompt_erase.py.

    modal run experiments/modal_prompt_erase.py --smoke
    modal run experiments/modal_prompt_erase.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-prompt-erase")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}
K_VALUES = (0, 1, 2, 4, 8)
# EXPLORATORY extension, logged as a deviation in PREREG_prompt_erase.md: the frozen
# matrix returned refit cv 1.000 at every k, so these ask where erasure bites at all.
K_EXPLORATORY = (16, 32, 48, 56)
ERASE_LAYERS = (26, 30)
PROBE_LAYER = 32
FRAMINGS = ("aversive", "neutral", "pleasant")
REMOTE_SRC = "/root/src"


@app.function(image=image, gpu="A100-40GB", timeout=14400, retries=0,
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
    from report_gap import directions as D
    from report_gap import hooks as H
    from report_gap import stimuli as S

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    model_name = PAIR[model_key]
    ks = K_VALUES[:3] if smoke else (K_VALUES + K_EXPLORATORY)
    layers = ERASE_LAYERS[:1] if smoke else ERASE_LAYERS
    n_items = 6 if smoke else None

    out_dir = "/data/pe_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pe.jsonl")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    hidden = model.config.hidden_size

    contexts = {f: S.build_prompt_induced(f, n_items) for f in FRAMINGS}
    n = len(contexts["neutral"])
    print("%s: %d items x %d framings" % (model_key, n, len(FRAMINGS)))

    def acts_at(texts, layer):
        return D.collect_activations(model, tok, texts, layer)

    # ---- the contrast the probe and the erasure basis are both fit on ----
    contrast_texts = contexts["aversive"] + contexts["pleasant"]
    contrast_y = np.array([1] * n + [0] * n)
    contrast_g = np.array(list(range(n)) + list(range(n)))     # topic-grouped CV

    # ---- the layer-32 probe: fit on CLEAN activations, never refit on erased ones ----
    probe = D.fit_direction(acts_at(contrast_texts, PROBE_LAYER), contrast_y, contrast_g,
                            layer=PROBE_LAYER)
    p32 = probe.vector / np.linalg.norm(probe.vector)
    p32_t = torch.tensor(p32, dtype=torch.float32).to("cuda")
    print("clean layer-%d probe cv %.3f" % (PROBE_LAYER, probe.cv_accuracy))

    records = []
    started = time.time()

    for E in layers:
        # ---- iterative nullspace basis at layer E, refitting on the projected activations ----
        A_e = acts_at(contrast_texts, E).astype(np.float64)
        work = A_e.copy()
        basis = []
        for step in range(max(ks)):
            d = D.fit_direction(work, contrast_y, contrast_g, layer=E)
            v = d.vector / np.linalg.norm(d.vector)
            for b in basis:                     # re-orthogonalize against what is already removed
                v = v - float(np.dot(v, b)) * b
            nv = np.linalg.norm(v)
            if nv < 1e-8:
                print("INLP step %d collapsed; stopping the basis at %d" % (step, len(basis)))
                break
            v = v / nv
            basis.append(v)
            work = work - np.outer(work @ v, v)
        basis = np.array(basis)
        print("[%6.1fs] L%d INLP basis rank %d" % (time.time() - started, E, len(basis)))

        rng = np.random.default_rng(0)
        rand_full = rng.normal(size=(max(ks), hidden))
        rand_full, _ = np.linalg.qr(rand_full.T)
        rand_full = rand_full.T

        for k in ks:
            for kind, B in (("fitted", basis[:k]), ("random", rand_full[:k])):
                if k == 0 and kind == "random":
                    continue                      # k=0 is the same no-op either way
                bt = torch.tensor(B, dtype=torch.float32).to("cuda") if k else torch.zeros(0, hidden)
                for framing in FRAMINGS:
                    texts = contexts[framing]
                    enc = tok(texts, return_tensors="pt", padding=True,
                              add_special_tokens=True).to("cuda")
                    with contextlib.ExitStack() as stack:
                        if k:
                            st = stack.enter_context(
                                H.project_out_subspace(model, E, bt))
                        with torch.no_grad():
                            out = model(**enc, output_hidden_states=True)
                    # +1: hidden_states[L] is the INPUT to layer L. `collect_activations`, which
                    # fit this probe, uses [layer + 1], so reading [PROBE_LAYER] here would apply a
                    # probe fit on layer 32's output to layer 31's output. Caught by the killer
                    # control returning identical numbers for fitted and random bases.
                    h32 = out.hidden_states[PROBE_LAYER + 1][:, -1, :].float()
                    hE = out.hidden_states[E + 1][:, -1, :].float()
                    reads = (h32 @ p32_t).tolist()
                    for i, text in enumerate(texts):
                        records.append({
                            "model_key": model_key, "erase_layer": E, "k": k, "kind": kind,
                            "framing": framing, "item_index": i, "cell": "%d|%s" % (i, framing),
                            "probe32": reads[i],
                            "hE": hE[i].tolist() if k in (0, max(ks)) else None,
                        })
            print("[%6.1fs] L%d k=%d done, %d records" % (time.time() - started, E, k,
                                                          len(records)))

        # ---- the erasure check: refit a probe ON THE ERASED activations at layer E ----
        # The erasure check is run under the FITTED basis and under a RANDOM one of the same
        # rank. Without the random arm the check cannot tell "the property was erased" from "the
        # sample was exhausted": with n samples in a d-dimensional stream, erasing k directions
        # eventually destroys separability for reasons that have nothing to do with the property.
        for k in [x for x in ks if x]:
            for kind, B in (("erasure_check", basis[:k]), ("erasure_check_random", rand_full[:k])):
                if len(B) < k:
                    continue
                with contextlib.ExitStack() as stack:
                    stack.enter_context(H.project_out_subspace(
                        model, E, torch.tensor(B, dtype=torch.float32).to("cuda")))
                    erased = acts_at(contrast_texts, E)
                refit = D.fit_direction(erased, contrast_y, contrast_g, layer=E)
                records.append({"model_key": model_key, "erase_layer": E, "k": k,
                                "kind": kind, "framing": "", "item_index": -1,
                                "cell": "%s|%d" % (kind, k), "refit_cv": refit.cv_accuracy,
                                "probe32": None, "hE": None})
                print("[%6.1fs] L%d k=%-3d %-21s refit cv %.3f"
                      % (time.time() - started, E, k, kind, refit.cv_accuracy))

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "erase_layers": list(layers),
                   "k_values": list(ks), "probe_layer": PROBE_LAYER,
                   "clean_probe32_cv": probe.cv_accuracy, "n_items": n,
                   "stimuli_sha256": S.frozen_hash("prompt_erase"), "injection": "none",
                   "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    data_vol.commit()
    return {"model": model_name, "model_key": model_key, "rows": len(records), "path": path,
            "clean_probe32_cv": probe.cv_accuracy, "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 80)
    print("PROMPT-INDUCED SUBSPACE ERASE%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 80)
    for res in results:
        print("%-30s rows=%6d clean probe32 cv %.3f  %.1f min"
              % (res["model"], res["rows"], res["clean_probe32_cv"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_prompt_erase.py on the artifacts.")
