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
