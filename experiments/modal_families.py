"""The confirmatory run for PREREG_families.md.

Every positive result in this repo rides on one architecture family. This runs the enumerate arm's
exact stimuli and exact format over a frozen list of matched base/instruct pairs across four
families, to find out whether the position prior is a property of preference tuning or a property of
Qwen2.5-3B-Instruct.

NO INJECTION ANYWHERE, and that is what makes the arm possible. `RESULTS.md` had to abandon Llama
because the valence direction was inert on it; a position prior needs no direction, no alpha and no
layer, so a model that cannot be steered can still be enumerated.

This runner COMPUTES NOTHING. It writes raw distributions and a per-model status record. Every gate,
endpoint and verdict lives in analyze_families.py, per the prereg.

    modal run experiments/modal_families.py --smoke     # 4 orderings, 4 items, 2 pairs
    modal run experiments/modal_families.py
    modal run experiments/modal_families.py --only qwen3b,llama3b
"""

from __future__ import annotations

import modal

app = modal.App("report-gap-families")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.12.1", "accelerate", "numpy", "sentencepiece")
    .add_local_dir("src", remote_path="/root/src")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("report-gap-data", create_if_missing=True)

# Frozen in PREREG_families.md section 3, in priority order. The run walks this list in order and
# stops at the budget cap, so a truncated run is truncated from the BOTTOM rather than from wherever
# it happened to be. Mirrors are used where upstream is gated: a run that needs an interactive
# licence click is not reproducible.
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

CONDITIONS = ("letters", "numbers", "identical", "canary")
N_ITEMS = 30
REMOTE_SRC = "/root/src"
PLAIN_TEMPLATE = "%s\n\n%s\nAnswer:"

# The existing artifact's stimuli hash. Asserted equal at run time so this arm provably consumes the
# same stimuli as the enumerate arm and its Qwen2.5-3B rows are a reproduction control rather than a
# differently-generated lookalike. Prereg section 7.
EXPECT_STIMULI_SHA = None  # filled from data/enum_instruct/header.json by the local entrypoint


@app.function(image=image, gpu="A100-40GB", timeout=14400, retries=0,
              volumes={"/root/.cache/huggingface": hf_cache, "/data": data_vol})
def run(spec: dict) -> dict:
    """One checkpoint, all 120 orderings, all four conditions. Streams and resumes."""
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
    smoke, expect_sha = spec["smoke"], spec.get("expect_stimuli_sha")
    tag = "%s_%s" % (pair_key, role)

    prov = report_gap.assert_provenance(expect_dir=os.path.join(REMOTE_SRC, "report_gap"))
    sha = S.frozen_hash("enumerate")
    if expect_sha and sha != expect_sha:
        raise RuntimeError(
            "stimuli hash %s != the enumerate arm's %s. This arm's whole comparability rests on "
            "consuming the SAME stimuli; a different hash means the reproduction control is "
            "meaningless." % (sha, expect_sha))

    orderings = S.all_option_orderings()
    if smoke:
        orderings = orderings[:4]
    n_items = 4 if smoke else N_ITEMS

    out_dir = "/data/fam_%s%s" % (tag, "_smoke" if smoke else "")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "enum.jsonl")
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

    # ---- load. A model that cannot be loaded is RECORDED, not silently dropped. ----
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda").eval()
    except Exception as exc:                                   # noqa: BLE001
        print("UNAVAILABLE %s: %s" % (model_name, exc))
        return write_status("unavailable", error="%s: %s" % (type(exc).__name__, exc),
                            traceback=traceback.format_exc()[-2000:], rows=0,
                            seconds=time.time() - started)

    # ---- label tokens. Per MODEL, not per project: a label that fuses with the leading space on
    # one tokenizer may not on another, and that bug already cost this project one arm. ----
    try:
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
                    raise RuntimeError(
                        "label %r maps to token %d decoding to %r, not the label"
                        % (L, tid, tok.decode([tid])))
            label_ids[L] = ids
        shared = [t_ for L in label_ids for t_ in label_ids[L]
                  if sum(t_ in v for v in label_ids.values()) > 1]
        if shared:
            raise RuntimeError("token(s) %s shared between labels" % sorted(set(shared)))
    except Exception as exc:                                   # noqa: BLE001
        print("UNAVAILABLE (tokens) %s: %s" % (model_name, exc))
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
                done.add((r["condition"], tuple(r["ordering"])))
        print("resuming: %d (condition, ordering) pair(s) already complete" % len(done))

    write_status("running", n_orderings=len(orderings), n_items=n_items,
                 conditions=list(CONDITIONS), injection="none",
                 label_tokens={L: label_ids[L] for L in sorted(label_ids)})

    prompts = S.build_prompts()[:n_items]
    written = 0

    for condition in CONDITIONS:
        labels = list(S.NUMBER_LABELS) if condition == "numbers" else list("ABCDE")
        for oi, ordering in enumerate(orderings):
            if (condition, ordering) in done:
                continue
            probe, mapping = S.build_enumerated_probe(ordering, condition)
            texts = [PLAIN_TEMPLATE % (p, probe) for p in prompts]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            with torch.no_grad():                    # no hook of any kind: no intervention here
                logits = model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)

            rows = []
            for i, prompt in enumerate(prompts):
                per = {L: float(max(probs[i, t] for t in label_ids[L])) for L in labels}
                off = 1.0 - sum(per.values())
                total = sum(per.values()) or 1.0
                norm = {L: v / total for L, v in per.items()}
                rows.append({
                    "pair_key": pair_key, "family": family, "role": role,
                    "model_key": tag, "condition": condition,
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
            if oi % 30 == 0:
                data_vol.commit()
                print("[%6.1fs] %-22s %-10s ordering %3d/%d rows=%d"
                      % (time.time() - started, tag, condition, oi + 1, len(orderings), written))
        data_vol.commit()

    return write_status("complete", rows=written, path=path, n_orderings=len(orderings),
                        n_items=n_items, seconds=time.time() - started)


@app.local_entrypoint()
def main(smoke: bool = False, only: str = ""):
    import json
    import pathlib

    # Assert the stimuli hash against the committed enumerate artifact, LOCALLY, before spending
    # anything. Prereg section 7.
    expect = None
    header = pathlib.Path("data/enum_instruct/header.json")
    if header.exists():
        expect = json.loads(header.read_text(encoding="utf-8")).get("stimuli_sha256")
        print("enumerate arm stimuli hash: %s" % expect)
    else:
        print("WARNING: no data/enum_instruct/header.json, cannot assert stimuli equality")

    wanted = {k.strip() for k in only.split(",") if k.strip()}
    specs = []
    for pair_key, family, base, instruct in PAIRS:
        if wanted and pair_key not in wanted:
            continue
        for role, model_name in (("base", base), ("instruct", instruct)):
            specs.append({"pair_key": pair_key, "family": family, "role": role,
                          "model": model_name, "smoke": smoke,
                          "expect_stimuli_sha": expect})
    if smoke and not wanted:
        specs = specs[:4]

    print("dispatching %d checkpoint(s)%s" % (len(specs), "  [SMOKE]" if smoke else ""))
    results = list(run.map(specs))

    print("\n" + "=" * 88)
    print("FAMILIES%s" % ("  [SMOKE]" if smoke else ""))
    print("=" * 88)
    ok = 0
    for res in results:
        if res["state"] == "complete":
            ok += 1
            print("%-24s %-10s %-34s rows=%6d  %.1f min"
                  % (res["pair_key"], res["role"], res["model"], res["rows"],
                     res["seconds"] / 60.0))
        else:
            print("%-24s %-10s %-34s %s: %s"
                  % (res["pair_key"], res["role"], res["model"], res["state"].upper(),
                     str(res.get("error"))[:80]))
    print("\n%d/%d checkpoints complete." % (ok, len(results)))
    print("Nothing is scored here. Run analyze_families.py on the artifacts.")
