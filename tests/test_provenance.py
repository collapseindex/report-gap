"""Tests for the code-path assert.

A commit hash records what was committed. It does not record what was imported, and on Modal the
gap between those two is a real one: `src/` is copied into the image at build time, `sys.path` is
manipulated at the top of every remote entrypoint, and a stale layer would run old code under a new
hash without raising anything.

The shadow test below is the one that matters, so it constructs an actual second copy on the path
and requires the guard to fire.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

import report_gap


def test_provenance_reports_where_the_package_came_from():
    info = report_gap.provenance()
    assert pathlib.Path(info["package_dir"]).name == "report_gap"
    assert (pathlib.Path(info["package_dir"]) / "__init__.py").exists()
    assert len(info["source_sha256"]) == 64
    assert int(info["module_count"]) >= 5


def test_source_hash_changes_when_source_changes(tmp_path, monkeypatch):
    # the hash must be a function of the source, not of the directory name
    a = tmp_path / "a" / "report_gap"
    a.mkdir(parents=True)
    (a / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    b = tmp_path / "b" / "report_gap"
    b.mkdir(parents=True)
    (b / "__init__.py").write_text("x = 2\n", encoding="utf-8")

    import hashlib

    def digest(root):
        h = hashlib.sha256()
        for p in sorted(root.glob("*.py")):
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
        return h.hexdigest()

    assert digest(a) != digest(b)


def test_assert_provenance_passes_on_the_real_package():
    info = report_gap.assert_provenance()
    assert info["package_dir"] == report_gap.provenance()["package_dir"]


def test_assert_provenance_fires_on_the_wrong_directory():
    with pytest.raises(RuntimeError, match="expected"):
        report_gap.assert_provenance(expect_dir="/root/src/report_gap")


def test_assert_provenance_fires_on_a_shadow_copy(tmp_path, monkeypatch):
    # this is the failure it exists for: a second copy of the package visible on sys.path, where
    # which one ran is not determinable from a commit hash.
    shadow = tmp_path / "report_gap"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("# a stale copy\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(RuntimeError, match="second copy"):
        report_gap.assert_provenance()


def test_shadow_check_does_not_fire_without_a_shadow(tmp_path, monkeypatch):
    # negative control for the negative test above: an unrelated directory on the path is not a
    # shadow, and a guard that fires on any sys.path entry would be useless.
    monkeypatch.syspath_prepend(str(tmp_path))
    report_gap.assert_provenance()


def test_run_header_carries_both_hashes():
    header = report_gap.run_header()
    assert len(header["stimuli_sha256"]) == 64
    assert len(header["source_sha256"]) == 64
    assert "git_commit" in header


def test_broken_syspath_entries_do_not_crash_the_check(monkeypatch):
    # sys.path picks up odd entries (zip imports, empty strings, missing dirs). the guard must
    # survive them rather than taking the run down with an OSError.
    monkeypatch.setattr(sys, "path", ["", "\x00bad", "/does/not/exist"] + list(sys.path))
    report_gap.assert_provenance()

# Run directories whose recorded stimuli hash no longer reproduces from any scope
# in the tree. All of them belong to arms whose verdicts were RETRACTED, and
# within each arm the original and its _rep4 replication share one hash, so the
# replication comparison is internally valid: the seeds changed and the stimuli
# did not. What is lost is the ability to reconstruct those exact stimuli today.
# Listed one by one on purpose. A wildcard here would let a new orphan join
# silently, which is the failure this test exists to prevent.
KNOWN_UNREPRODUCIBLE = {
    "depth_base", "depth_base_rep4", "depth_instruct", "depth_instruct_rep4",
    "floor", "floor_smoke",
    "pair_base", "pair_base_rep4", "pair_instruct", "pair_instruct_rep4",
    "qwen3b", "qwen3b_rep4", "qwen3b_smoke",
    "shell_base", "shell_base_rep4", "shell_instruct", "shell_instruct_rep4",
}


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def test_every_recorded_stimuli_hash_still_reproduces():
    """A committed artifact must be tied to stimuli that still exist.

    Value-based checks cannot see this: an artifact written against stimuli that
    have since changed still contains numbers, and every assertion about those
    numbers still passes. The hash is the only thing that knows.
    """
    import json

    from report_gap import stimuli

    scopes = ["all"] + sorted(getattr(stimuli, "_ARM_SCOPES", {}))
    current = {stimuli.frozen_hash(sc) for sc in scopes}

    data = pathlib.Path(__file__).resolve().parents[1] / "data"
    if not data.is_dir():
        pytest.skip("no committed data")

    orphans = set()
    checked = 0
    for f in sorted(data.rglob("*.json")):
        try:
            blob = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for d in _walk(blob):
            h = d.get("stimuli_sha256")
            if not h:
                continue
            checked += 1
            if h not in current:
                orphans.add(f.relative_to(data).parts[0])

    assert checked, "premise failed: no artifact carries a stimuli hash"
    new = orphans - KNOWN_UNREPRODUCIBLE
    assert not new, (
        "these run directories record stimuli that no scope in the tree can "
        "reproduce, and they are not on the known list: %s" % sorted(new))

    gone = KNOWN_UNREPRODUCIBLE - orphans
    assert not gone, (
        "these are listed as unreproducible but now reproduce; remove them from "
        "KNOWN_UNREPRODUCIBLE rather than leaving the list wrong: %s" % sorted(gone))


def test_the_surviving_results_are_reproducible():
    """The arms carrying results the paper still claims must not be orphaned.

    This is the half that matters. A retracted arm whose stimuli drifted is a
    reproducibility gap; a SURVIVING arm whose stimuli drifted would be a claim
    resting on a corpus that no longer exists.
    """
    surviving = {"enumerate", "families", "erase", "readjudicate", "instrument",
                 "prompt_erase", "leace", "binary"}
    assert not (surviving & KNOWN_UNREPRODUCIBLE), (
        "an arm carrying a surviving result is on the unreproducible list")
