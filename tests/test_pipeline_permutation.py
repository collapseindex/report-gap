"""The permutation test on the ANALYSIS PIPELINE, not on the data.

Shuffle which condition each cell's rows are labelled with, preserving every other structure, and
rerun an analyzer. A correct pipeline must go null: a pipeline that finds structure in shuffled
labels is finding it in the labels.

This project shipped five checker defects, every one of which failed in the flattering direction
(a headline check that printed "write the sentence" on a refuted claim, a capability gate that
passed on +0.0000, a preregistered contrast with an inverted sign, a verdict branch that called
"both moved" CORE-ABSENT, and a profile computed over a gate-failed layer). This is the test that
catches that whole class, and it did not exist until after all five had happened.

Skips if the artifacts are absent, so the suite still runs on a clean checkout.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import random
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ERASE_BASE = ROOT / "data" / "erase_base" / "erase.jsonl"
ERASE_INST = ROOT / "data" / "erase_instruct" / "erase.jsonl"


def _load_analyzer(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / ("%s.py" % name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shuffle_condition_labels(src, dst, seed=0):
    """Permute condition labels within each (cell, erase_layer) group. Nothing else changes."""
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["cell"], r["erase_layer"])].append(r)
    for g in groups.values():
        conds = [r["condition"] for r in g]
        rng.shuffle(conds)
        for r, c in zip(g, conds):
            r["condition"] = c
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return len(rows)


def _sandbox_base(tmp_path):
    """Copy the base artifact into tmp_path too, so the analyzer's output lands there.

    The analyzer names its verdict file after argv[1]'s grandparent. Handing it the real
    `data/erase_base/...` makes every test run drop a timestamped verdict into `data/`, including
    one scored on SHUFFLED labels, which is indistinguishable from a real run's output once it is
    sitting in the artifact directory. Tests do not write into `data/`.
    """
    dst = tmp_path / "base" / "erase.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ERASE_BASE, dst)
    shutil.copy(ERASE_BASE.parent / "ordering_variance.json",
                dst.parent / "ordering_variance.json")
    return dst


@pytest.mark.skipif(not ERASE_INST.exists(), reason="erase artifacts not present")
def test_erase_analyzer_goes_null_on_shuffled_labels(tmp_path, capsys):
    dst = tmp_path / "shuffled" / "erase.jsonl"
    n = _shuffle_condition_labels(ERASE_INST, dst)
    assert n > 0
    shutil.copy(ERASE_INST.parent / "ordering_variance.json",
                dst.parent / "ordering_variance.json")

    analyze = _load_analyzer("analyze_erase")
    analyze.main(["analyze_erase.py", str(_sandbox_base(tmp_path)), str(dst)])
    out = capsys.readouterr().out

    assert "VERDICT: NO_INSTRUMENT" in out, (
        "the erase analyzer found a verdict in label-shuffled data. It is finding structure in the "
        "labels, not the measurements.\n\n%s" % out[-2000:]
    )


@pytest.mark.skipif(not ERASE_INST.exists(), reason="erase artifacts not present")
def test_the_shuffle_actually_shuffles(tmp_path):
    """Negative control for the test above: if the shuffle were a no-op it would pass vacuously."""
    dst = tmp_path / "shuffled" / "erase.jsonl"
    _shuffle_condition_labels(ERASE_INST, dst)
    before = [json.loads(l)["condition"]
              for l in ERASE_INST.read_text(encoding="utf-8").splitlines() if l.strip()]
    after = [json.loads(l)["condition"]
             for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert before != after, "the shuffle changed nothing"
    assert collections.Counter(before) == collections.Counter(after), \
        "the shuffle changed the condition COUNTS, so it is not a permutation"


@pytest.mark.skipif(not ERASE_INST.exists(), reason="erase artifacts not present")
def test_the_real_artifact_still_yields_a_verdict(tmp_path, capsys):
    """The other half: a test that only checks shuffled data would pass on a dead analyzer."""
    analyze = _load_analyzer("analyze_erase")
    analyze.main(["analyze_erase.py", str(_sandbox_base(tmp_path)), str(ERASE_INST)])
    out = capsys.readouterr().out
    assert "VERDICT: NO_INSTRUMENT" not in out, \
        "the analyzer returns NO_INSTRUMENT on the real artifact too, so the shuffle test above " \
        "proves nothing about its sensitivity"
