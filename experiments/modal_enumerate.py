"""The confirmatory run for PREREG_enumerate.md.

RESULTS_replication.md killed three verdicts because four sampled option orderings were not enough
to average out order effects. There are only 120 orderings. This runs all of them, so the nuisance
stops being a sampled quantity and becomes a measured population.

NO INJECTION ANYWHERE. Every cell is a plain forward pass with no hook attached, which is why the
output can be stated as a population fact rather than an effect. Four conditions:

    letters    the real five options, A-E. The quantity every previous arm measured.
    numbers    the real five options, 1-5. Is it the label alphabet?
    identical  the SAME sentence five times. Pure position prior, zero content. The denominator.
    canary     "which of these is the number four". Known answer, no self-report content, so
               "the instrument is order-sensitive" separates from "self-report is".

    modal run experiments/modal_enumerate.py
    modal run experiments/modal_enumerate.py --smoke
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-enumerate")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "numpy", "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

PAIR = {"base": "Qwen/Qwen2.5-3B", "instruct": "Qwen/Qwen2.5-3B-Instruct"}
CONDITIONS = ("letters", "numbers", "identical", "canary")
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import report_gap
    from report_gap import analysis as A
    from report_gap import hooks as H
    from report_gap import stimuli as S

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    model_name = PAIR[model_key]
    orderings = S.all_option_orderings()
    if smoke:
        orderings = orderings[:4]
    n_items = 4 if smoke else N_ITEMS

    out_dir = "/data/enum_%s%s" % (model_key, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "enum.jsonl")

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["condition"], tuple(r["ordering"])))
        print("resuming: %d batch(es) already complete" % len(done))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda").eval()

    prompts = S.build_prompts()[:n_items]
    # Taking c[0] of the space-prefixed encoding is WRONG for labels that do not fuse with the
    # space. On Qwen, encode(" A") == [362] (one token) but encode(" 1") == [220, 16]: a space
    # token then the digit. c[0] would read token 220 for EVERY digit, giving all five labels the
    # same probability and a renormalized 0.2 each. That is exactly what the first run of this arm
    # produced, with off_option_mass at -3.99.
    #
    # Take the LAST token of each encoding, which is the one that carries the label, and assert it
    # decodes back to the label so the bug cannot recur silently.
    label_ids = {}
    for L in list("ABCDE") + list(S.NUMBER_LABELS):
        ids = []
        for enc_ in (tok.encode(L, add_special_tokens=False),
                     tok.encode(" " + L, add_special_tokens=False)):
            if enc_:
                ids.append(enc_[-1])
        ids = sorted(set(ids))
        if not ids:
            raise RuntimeError("label %r has no token form" % L)
        for tid in ids:
            if tok.decode([tid]).strip() != L:
                raise RuntimeError("label %r maps to token %d which decodes to %r, not the label. "
                                   "Reading it would score a different token entirely."
                                   % (L, tid, tok.decode([tid])))
        label_ids[L] = ids
    shared = [t_ for L in label_ids for t_ in label_ids[L]
              if sum(t_ in v for v in label_ids.values()) > 1]
    if shared:
        raise RuntimeError("token(s) %s are shared between labels; every label would read the "
                           "same mass" % sorted(set(shared)))
    print("label tokens: %s" % {L: label_ids[L] for L in sorted(label_ids)})

    header = dict(prov)
    header.update({"model": model_name, "model_key": model_key,
                   "stimuli_sha256": S.frozen_hash("enumerate"),
                   "n_orderings": len(orderings), "n_items": n_items,
                   "conditions": list(CONDITIONS), "injection": "none", "smoke": smoke})
    with open(os.path.join(out_dir, "header.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=1)
    print(json.dumps({k: v for k, v in header.items() if k != "package_dir"}, indent=1))

    started, written = time.time(), 0

    for condition in CONDITIONS:
        labels = list(S.NUMBER_LABELS) if condition == "numbers" else list("ABCDE")
        for oi, ordering in enumerate(orderings):
            if (condition, ordering) in done:
                continue
            probe, mapping = S.build_enumerated_probe(ordering, condition)
            texts = [PLAIN_TEMPLATE % (p, probe) for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            # no hook of any kind: this arm has no intervention
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)

            rows = []
            for i, prompt in enumerate(prompts):
                per = {L: float(max(probs[i, t] for t in label_ids[L])) for L in labels}
                off = 1.0 - sum(per.values())
                total = sum(per.values()) or 1.0
                norm = {L: v / total for L, v in per.items()}
                rows.append({
                    "model_key": model_key, "condition": condition,
                    "ordering": list(ordering), "ordering_index": oi,
                    "item": prompt, "cell": "%s|%d" % (prompt, oi),
                    "mapping": mapping, "probs": norm, "off_option_mass": off,
                    "argmax": max(norm, key=norm.get),
                    "entropy": A.option_entropy(norm),
                })
            with open(path, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            written += len(rows)
            if oi % 20 == 0:
                data_vol.commit()
                print("[%6.1fs] %-10s ordering %3d/%d  rows=%d"
                      % (time.time() - started, condition, oi + 1, len(orderings), written))
        data_vol.commit()
        print("[%6.1fs] %s complete" % (time.time() - started, condition))

    return {"model": model_name, "rows": written, "path": path,
            "orderings": len(orderings), "seconds": time.time() - started}


@app.local_entrypoint()
def main(smoke: bool = False):
    results = list(run.map(sorted(PAIR), kwargs={"smoke": smoke}))
    print("\n" + "=" * 78)
    print("ENUMERATION%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 78)
    for res in results:
        print("%-30s rows=%6d  orderings=%d  %.1f min"
              % (res["model"], res["rows"], res["orderings"], res["seconds"] / 60.0))
    print("\nNothing is scored here. Run analyze_enumerate.py on the artifacts.")
