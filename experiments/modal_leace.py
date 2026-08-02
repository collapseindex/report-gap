"""The confirmatory run for PREREG_leace.md.

Two changes from modal_prompt_erase.py, and only two: LEACE instead of INLP, and an erasure check
read on HELD-OUT items so it cannot confirm itself.

The eraser is fit on a train split of TOPICS and every endpoint is read on held-out topics. The
layer-32 probe is fit once on clean train activations and never refit on erased data. The random
control is rank-matched to LEACE (rank one), so the comparison is erasure against erasure rather
than erasure against dimensionality.

NO INJECTION ANYWHERE. This runner COMPUTES NOTHING beyond the activations and probe reads it
records; every gate and endpoint lives in analyze_leace.py.

    modal run experiments/modal_leace.py --smoke
    modal run experiments/modal_leace.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-leace")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "scikit-learn", "numpy",
                 "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}
ERASE_LAYERS = (26, 30)
PROBE_LAYER = 32
FRAMINGS = ("aversive", "neutral", "pleasant")
TRAIN_TOPICS = 20          # of 30; the remaining 10 are held out, split BY TOPIC
BATCH = 32
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
    from report_gap.leace import class_mean_gap, fit_leace

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    model_name = PAIR[model_key]
    layers = ERASE_LAYERS[:1] if smoke else ERASE_LAYERS
    n_topics = 6 if smoke else None
    n_train = 4 if smoke else TRAIN_TOPICS

    out_dir = "/data/leace_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "leace.jsonl")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    hidden = model.config.hidden_size

    # ---- contexts, and the SPLIT BY TOPIC ----
    contexts = {f: S.build_prompt_induced_large(f, n_topics) for f in FRAMINGS}
    topics = S.REVIEW_CONTEXTS if n_topics is None else S.REVIEW_CONTEXTS[:n_topics]
    per_topic = len(contexts["neutral"]) // len(topics)
    topic_of = [i // per_topic for i in range(len(contexts["neutral"]))]
    train_mask = np.array([t < n_train for t in topic_of])
    assert not (set(np.array(topic_of)[train_mask]) & set(np.array(topic_of)[~train_mask])), \
        "a topic appears on both sides of the split; the held-out set is not held out"
    print("%s: %d contexts per framing, %d topics, train %d / held-out %d items"
          % (model_key, len(contexts["neutral"]), len(topics),
             int(train_mask.sum()), int((~train_mask).sum())))

    def acts(texts, layer, ctx=None):
        out = []
        for s in range(0, len(texts), BATCH):
            enc = tok(texts[s:s + BATCH], return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with contextlib.ExitStack() as stack:
                if ctx is not None:
                    stack.enter_context(ctx())
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1]
            out.append(hs[:, -1, :].float().cpu().numpy())
        return np.concatenate(out, axis=0)

    contrast = contexts["aversive"] + contexts["pleasant"]
    y = np.array([1] * len(contexts["aversive"]) + [0] * len(contexts["pleasant"]))
    tr = np.concatenate([train_mask, train_mask])
    grp = np.concatenate([np.array(topic_of), np.array(topic_of)])

    # ---- the layer-32 probe: fit on CLEAN TRAIN only, never refit ----
    a32 = acts(contrast, PROBE_LAYER)
    probe = D.fit_direction(a32[tr], y[tr], grp[tr], layer=PROBE_LAYER)
    p32 = probe.vector / np.linalg.norm(probe.vector)
    print("clean layer-%d probe, fit on train only: cv %.3f" % (PROBE_LAYER, probe.cv_accuracy))

    records, started = [], time.time()
    rng = np.random.default_rng(0)

    for E in layers:
        aE = acts(contrast, E)
        er = fit_leace(aE[tr], y[tr])                    # fit on TRAIN items only
        assert er.rank == 1, "LEACE should be rank 1 for a binary label, got %d" % er.rank

        # rank-matched random affine eraser: remove one random whitened-space direction
        v = rng.normal(size=hidden)
        v /= np.linalg.norm(v)
        rand_proj = np.eye(hidden) - np.outer(v, v)
        erasers = {
            "leace": (er.proj, er.bias),
            "random": (rand_proj, aE[tr].mean(axis=0)),
        }
        print("[%6.1fs] L%d LEACE fit on train: rank %d, train class-mean gap %.3e -> %.3e"
              % (time.time() - started, E, er.rank,
                 class_mean_gap(aE[tr], y[tr]), class_mean_gap(er(aE[tr]), y[tr])))

        # Assert the hook reaches the index the erasure check reads. Without this, an off-by-one
        # makes the gate unfalsifiable, which is exactly what happened before.
        probe_clean = acts(contrast[:BATCH], E + 1)
        probe_dirty = acts(contrast[:BATCH], E + 1,
                           lambda: H.apply_affine(model, E,
                                                  torch.zeros(hidden, hidden),
                                                  torch.zeros(hidden)))
        moved = float(np.abs(probe_clean - probe_dirty).max())
        if moved < 1e-3:
            raise RuntimeError(
                "a hook that zeroes layer %d does not change the erasure-check read; the read "
                "index is upstream of the hook and the gate cannot fail" % E)
        print("[%6.1fs] L%d hook reaches the check read (zero-hook moves it by %.1f)"
              % (time.time() - started, E, moved))

        for kind, (proj, bias) in list(erasers.items()) + [("clean", (None, None))]:
            def ctx(proj=proj, bias=bias):
                return H.apply_affine(model, E,
                                      torch.tensor(proj, dtype=torch.float32),
                                      torch.tensor(bias, dtype=torch.float32))
            hook = None if kind == "clean" else ctx

            # The erasure check, read on HELD-OUT items at E+2.
            #
            # E+2, not E+1. Verified empirically on this model: a hook on layer E that ZEROES the
            # entire stream leaves hidden_states[E] and hidden_states[E+1] untouched and first
            # changes hidden_states[E+2]. Reading at E+1 reads UPSTREAM of the hook, which is why
            # this check returned cv 1.000 in every previous arm: it was measuring un-erased
            # activations and could not fail.
            eE = acts(contrast, E + 1, hook)
            records.append({
                "model_key": model_key, "erase_layer": E, "kind": kind, "what": "erasure_check",
                "gap_heldout": float(class_mean_gap(eE[~tr], y[~tr])),
                "acts_heldout": None,
                "refit_cv": float(D.fit_direction(eE[~tr], y[~tr], grp[~tr], layer=E).cv_accuracy),
            })

            # the primary: clean-fit layer-32 probe on erased held-out activations
            for framing in FRAMINGS:
                reads = acts(contexts[framing], PROBE_LAYER, hook) @ p32
                for i, r in enumerate(reads):
                    if train_mask[i]:
                        continue
                    records.append({
                        "model_key": model_key, "erase_layer": E, "kind": kind,
                        "what": "probe32", "framing": framing, "item_index": int(i),
                        "topic": int(topic_of[i]), "read": float(r),
                    })
            print("[%6.1fs] L%d %-7s done, %d records" % (time.time() - started, E, kind,
                                                          len(records)))

    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key, "erase_layers": list(layers),
                   "probe_layer": PROBE_LAYER, "train_topics": n_train,
                   "n_contexts_per_framing": len(contexts["neutral"]),
                   "clean_probe32_cv_train": probe.cv_accuracy,
                   "stimuli_sha256": S.frozen_hash("prompt_erase"),
                   "injection": "none", "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    data_vol.commit()
    return {"model": model_name, "model_key": model_key, "rows": len(records),
            "clean_probe32_cv": probe.cv_accuracy, "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 80)
    print("LEACE ERASURE%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 80)
    for res in results:
        print("%-30s rows=%6d  clean probe32 cv(train) %.3f  %.1f min"
              % (res["model"], res["rows"], res["clean_probe32_cv"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_leace.py on the artifacts.")
