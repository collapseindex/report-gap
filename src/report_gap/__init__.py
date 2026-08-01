"""report_gap: judge-free measurement of what a forced-choice readout loses.

`provenance()` exists because a commit hash records what was committed, not what was imported. The
Modal image copies `src/` in at build time, `sys.path` is manipulated at the top of every remote
entrypoint, and a stale layer or a second copy on the path would run old code under a new commit
hash without a single error. `CONTROLS.md` section 8 calls this establishing the source of truth for
the code path: the artifact records where the module actually came from, so the question is
answerable afterwards instead of assumed.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

__all__ = ["provenance", "assert_provenance"]


def provenance() -> dict[str, str]:
    """Where this package was imported from, and what its source hashes to.

    Returns:
        A dict with the package directory, the module count, and a SHA-256 over the sorted
        contents of every `.py` file in the package. Written into every artifact.
    """
    root = pathlib.Path(__file__).resolve().parent
    files = sorted(p for p in root.glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "package_dir": str(root),
        "module_count": str(len(files)),
        "source_sha256": digest.hexdigest(),
    }


def assert_provenance(expect_dir: str | None = None) -> dict[str, str]:
    """Assert the imported package is the one the run intends, and return its provenance.

    Called at the top of every remote entrypoint, before any model is loaded, so a run against a
    stale image fails immediately rather than producing scorable numbers from unknown code.

    Args:
        expect_dir: Directory the package is expected to live in, if the caller knows it. On Modal
            this is the mount point the image was built with.

    Returns:
        The same dict as `provenance()`.

    Raises:
        RuntimeError: If the package resolved somewhere other than `expect_dir`, or if a second
            copy of it is visible earlier on `sys.path`, either of which means the code that ran is
            not the code that was read.
    """
    import sys

    info = provenance()
    actual = pathlib.Path(info["package_dir"]).resolve()

    if expect_dir is not None and actual != pathlib.Path(expect_dir).resolve():
        raise RuntimeError("report_gap imported from %s, expected %s" % (actual, expect_dir))

    shadows = []
    for entry in sys.path:
        try:
            candidate = (pathlib.Path(entry) / "report_gap" / "__init__.py").resolve()
        except (OSError, ValueError):
            continue
        if candidate.exists() and candidate.parent != actual:
            shadows.append(str(candidate.parent))
    if shadows:
        raise RuntimeError("a second copy of report_gap is on sys.path at %s; the imported one is "
                           "%s and which one ran is not determinable from a commit hash"
                           % (", ".join(sorted(set(shadows))), actual))

    return info


# convenience for artifact headers, so a run records the environment it actually got
def run_header() -> dict[str, str]:
    """Provenance plus the environment fields every artifact carries.

    Returns:
        Dict suitable for writing at the top of a results file.
    """
    from . import stimuli

    info = provenance()
    info["stimuli_sha256"] = stimuli.frozen_hash()
    info["git_commit"] = os.environ.get("REPORT_GAP_COMMIT", "unset")
    return info
