"""The confirmatory run for PREREG_instrument.md.

Turns the position prior from a nuisance into the object of study:

  determinacy          6 item types x 120 orderings x 3 paraphrases. Position dominance against a
                       determinacy axis measured independently of position, so the two are not an
                       identity.
  introspect_forward   "how much does option ORDER affect which option you pick?", 120 orderings.
  introspect_reverse   the same question worded inversely. Acquiescence control.
  introspect_placebo   the same shape, asking about the PHASE OF THE MOON. Format-artifact control.

The introspection probe is a five-option forced choice and therefore subject to the exact bias it
asks about, which is why every ordering is run and the analyzer reads it marginalized. Measuring a
belief about position at ONE ordering would be the error this project documents.

NO INJECTION ANYWHERE. This runner COMPUTES NOTHING; every gate and endpoint is in
analyze_instrument.py.

    modal run experiments/modal_instrument.py --smoke
    modal run experiments/modal_instrument.py
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-instrument")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "numpy", "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

# Same 16 checkpoints as PREREG_families.md, unchanged, so the Q2 correlation can be taken against
# position priors already measured rather than re-measured here.
PAIRS = [
    ("qwen3b",    "Qwen2.5",   "Qwen/Qwen2.5-3B",            "Qwen/Qwen2.5-3B-Instruct"),
    ("llama3b",   "Llama-3.2", "unsloth/Llama-3.2-3B",       "unsloth/Llama-3.2-3B-Instruct"),
    ("qwen1_5b",  "Qwen2.5",   "Qwen/Qwen2.5-1.5B",          "Qwen/Qwen2.5-1.5B-Instruct"),
    ("qwen7b",    "Qwen2.5",   "Qwen/Qwen2.5-7B",            "Qwen/Qwen2.5-7B-Instruct"),
    ("llama1b",   "Llama-3.2", "unsloth/Llama-3.2-1B",       "unsloth/Llama-3.2-1B-Instruct"),
    ("gemma2b",   "Gemma-2",   "unsloth/gemma-2-2b",         "unsloth/gemma-2-2b-it"),
    ("mistral7b", "Mistral",   "unsloth/mistral-7b-v0.3",    "unsloth/mistral-7b-instruct-v0.3"),
    ("qwen0_5b",  "Qwen2.5",   "Qwen/Qwen2.5-0.5B",          "Qwen/Qwen2.5-0.5B-Instruct"),
]

REMOTE_SRC = "/root/src"


@app.function(image=image, gpu="A100-40GB", timeout=14400, retries=0,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(spec: dict) -> dict:
    import json
    import os
    import sys
    import time
    import traceback

    sys.path.insert(0, REMOTE_SRC)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import report_gap
    from report_gap import analysis as A
    from report_gap import stimuli as S

    pair_key, family, role, model_name = (
        spec["pair_key"], spec["family"], spec["role"], spec["model"])
    smoke = spec["smoke"]
    tag = "%s_%s" % (pair_key, role)

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    sha = S.frozen_hash("instrument")

    orderings = S.all_option_orderings()
    if smoke:
        orderings = orderings[:5]
    paraphrases = range(1 if smoke else len(S.DETERMINACY_PARAPHRASES))
    items = [k for k, _, _, _ in S.DETERMINACY_BATTERY]
    if smoke:
        items = items[:2]

    out_dir = "/data/instr_%s%s" % (tag, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "instr.jsonl")
    status_path = os.path.join(out_dir, "status.json")

    def write_status(state, **extra):
        rec = {"pair_key": pair_key, "family": family, "role": role, "model": model_name,
               "state": state, "stimuli_sha256": sha, "smoke": smoke}
        rec.update(prov)
        rec.update(extra)
        with open(status_path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        data_vol.commit()
        return rec

    started = time.time()

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    except Exception as exc:                                   # noqa: BLE001
        return write_status("unavailable", error="%s: %s" % (type(exc).__name__, exc),
                            traceback=traceback.format_exc()[-2000:], rows=0,
                            seconds=time.time() - started)

    try:
        label_ids = {}
        for L in "ABCDE":
            ids = sorted({enc[-1] for enc in (tok.encode(L, add_special_tokens=False),
                                              tok.encode(" " + L, add_special_tokens=False))
                          if enc})
            if not ids:
                raise RuntimeError("label %r has no token form" % L)
            for tid in ids:
                if tok.decode([tid]).strip() != L:
                    raise RuntimeError("label %r maps to token %d decoding to %r"
                                       % (L, tid, tok.decode([tid])))
            label_ids[L] = ids
        shared = [t for L in label_ids for t in label_ids[L]
                  if sum(t in v for v in label_ids.values()) > 1]
        if shared:
            raise RuntimeError("token(s) %s shared between labels" % sorted(set(shared)))
    except Exception as exc:                                   # noqa: BLE001
        return write_status("unavailable", error="label tokens: %s" % exc, rows=0,
                            seconds=time.time() - started)

    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["condition"], r.get("item", ""), r.get("paraphrase", -1),
                          tuple(r["ordering"])))

    write_status("running", n_orderings=len(orderings), items=items,
                 conditions=["determinacy", "introspect_forward", "introspect_reverse",
                             "introspect_placebo"], injection="none")

    def score(texts, metas):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to("cuda")
        with torch.no_grad():                       # no hook of any kind
            logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        out = []
        for i, meta in enumerate(metas):
            per = {L: float(max(probs[i, t] for t in label_ids[L])) for L in "ABCDE"}
            off = 1.0 - sum(per.values())
            total = sum(per.values()) or 1.0
            norm = {L: v / total for L, v in per.items()}
            rec = dict(meta)
            rec.update({"pair_key": pair_key, "family": family, "role": role, "model_key": tag,
                        "probs": norm, "off_option_mass": off,
                        "argmax": max(norm, key=norm.get), "entropy": A.option_entropy(norm)})
            out.append(rec)
        return out

    written = 0

    def flush(rows):
        nonlocal written
        if not rows:
            return
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        written += len(rows)

    # ---- determinacy: batch the three paraphrases together at each (item, ordering) ----
    for item in items:
        for oi, ordering in enumerate(orderings):
            texts, metas = [], []
            for pi in paraphrases:
                if ("determinacy", item, pi, ordering) in done:
                    continue
                probe, mapping, correct = S.build_determinacy_probe(item, ordering, pi)
                texts.append(probe)
                metas.append({"condition": "determinacy", "item": item, "paraphrase": pi,
                              "ordering": list(ordering), "ordering_index": oi,
                              "mapping": mapping, "correct": correct})
            if texts:
                flush(score(texts, metas))
        data_vol.commit()
        print("[%6.1fs] %-22s determinacy %-14s rows=%d"
              % (time.time() - started, tag, item, written))

    # ---- introspection: batch the three variants together at each ordering ----
    for oi, ordering in enumerate(orderings):
        texts, metas = [], []
        for variant in ("forward", "reverse", "placebo"):
            cond = "introspect_%s" % variant
            if (cond, "", -1, ordering) in done:
                continue
            probe, mapping = S.build_introspection_probe(ordering, variant)
            texts.append(probe)
            metas.append({"condition": cond, "item": "", "paraphrase": -1,
                          "ordering": list(ordering), "ordering_index": oi,
                          "mapping": mapping, "correct": None})
        if texts:
            flush(score(texts, metas))
        if oi % 40 == 0:
            data_vol.commit()
            print("[%6.1fs] %-22s introspection ordering %3d/%d rows=%d"
                  % (time.time() - started, tag, oi + 1, len(orderings), written))
    data_vol.commit()

    return write_status("complete", rows=written, path=path, n_orderings=len(orderings),
                        seconds=time.time() - started)


@app.local_entrypoint()
def main(smoke: bool = False, only: str = ""):
    wanted = {k.strip() for k in only.split(",") if k.strip()}
    specs = []
    for pair_key, family, base, instruct in PAIRS:
        if wanted and pair_key not in wanted:
            continue
        for role, model_name in (("base", base), ("instruct", instruct)):
            specs.append({"pair_key": pair_key, "family": family, "role": role,
                          "model": model_name, "smoke": smoke})
    if smoke and not wanted:
        specs = specs[:2]

    print("dispatching %d checkpoint(s)%s" % (len(specs), "  [SMOKE]" if smoke else ""))
    results = list(run.map(specs))

    print("\n" + "=" * 88)
    print("INSTRUMENT%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 88)
    ok = 0
    for res in results:
        if res["state"] == "complete":
            ok += 1
            print("%-24s %-9s %-34s rows=%6d  %.1f min"
                  % (res["pair_key"], res["role"], res["model"], res["rows"],
                     res["seconds"] / 60.0))
        else:
            print("%-24s %-9s %-34s %s: %s"
                  % (res["pair_key"], res["role"], res["model"], res["state"].upper(),
                     str(res.get("error"))[:80]))
    print("\n%d/%d checkpoints complete." % (ok, len(results)))
    print("Nothing is scored here. Run analyze_instrument.py on the artifacts.")
