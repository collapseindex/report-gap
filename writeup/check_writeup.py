"""Structural checks on main.tex that do not require a LaTeX toolchain.

Every check here corresponds to a way this paper could ship broken without anyone noticing at
draft stage: a \\ref to a label that does not exist renders as "??", a \\cite to a missing bib key
renders as "[?]", and a number in the prose that disagrees with the repository is the failure this
whole project is about.

    python check_writeup.py
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def check_refs(tex: str) -> list[str]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    dangling = sorted(refs - labels)
    print("  labels %d, refs %d, dangling %s" % (len(labels), len(refs), dangling or "none"))
    return ["dangling ref: %s" % r for r in dangling]


def check_cites(tex: str) -> list[str]:
    raw = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex)
    cited = {c.strip() for group in raw for c in group.split(",")}
    bib = (HERE / "refs.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing = sorted(cited - keys)
    unused = sorted(keys - cited)
    print("  cites %d, bib keys %d, missing %s, unused %s"
          % (len(cited), len(keys), missing or "none", unused or "none"))
    return ["cite with no bib entry: %s" % k for k in missing]


def check_style(tex: str) -> list[str]:
    """The repo bans em dashes and double hyphens used as punctuation."""
    problems = []
    for i, line in enumerate(tex.splitlines(), 1):
        if "\u2014" in line or "\u2013" in line:
            problems.append("line %d: em/en dash" % i)
    print("  em/en dashes: %s" % (problems or "none"))
    return problems


def check_numbers(tex: str) -> list[str]:
    """Spot-check load-bearing numbers in the prose against the committed artifacts.

    A paper about a project whose headline died to an unchecked number should not itself carry
    unchecked numbers. These are the ones a reader would quote.
    """
    problems = []
    checks = []

    enum = ROOT / "data" / "enum_instruct" / "enum.jsonl"
    if enum.exists():
        rows = [json.loads(l) for l in enum.read_text(encoding="utf-8").splitlines() if l.strip()]
        ident = [r for r in rows if r["condition"] == "identical"]
        if ident:
            a = sum(r["probs"]["A"] for r in ident) / len(ident)
            checks.append(("0.8725", a, "identical-options mass on label A"))

    # the headline ratio, which the prose and the figure must both floor the same way
    if enum.exists():
        import collections
        per = collections.defaultdict(list)
        for r in rows:
            if r["condition"] == "letters":
                per[tuple(r["ordering"])].append(
                    sum(v for L, v in r["probs"].items()
                        if r["mapping"][L] in ("neg1", "neg2")))
        means = sorted(statistics.fmean(v) for v in per.values())
        checks.append(("986", float(int(means[-1] / means[0])),
                       "ordering ratio, floored (prose and figure must agree)"))
        checks.append(("0.0009", round(means[0], 4), "minimum ordering mass"))
        checks.append(("0.8820", round(means[-1], 4), "maximum ordering mass"))

    erase = ROOT / "data" / "erase_instruct" / "erase.jsonl"
    if erase.exists():
        rows = [json.loads(l) for l in erase.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = len({r["cell"] for r in rows})
        checks.append((None, n, "erase arm distinct cells (informational)"))

    for claimed, actual, what in checks:
        if claimed is None:
            print("  %-46s actual %s" % (what, actual))
            continue
        present = claimed in tex
        agrees = abs(float(claimed) - actual) < 5e-4
        print("  %-46s claims %s, artifact %.4f, in tex: %s, agrees: %s"
              % (what, claimed, actual, present, agrees))
        if present and not agrees:
            problems.append("%s: tex says %s, artifact says %.4f" % (what, claimed, actual))
    return problems


def main() -> int:
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    print("checking main.tex (%d lines)" % len(tex.splitlines()))
    problems = []
    print("\ncross-references")
    problems += check_refs(tex)
    print("\ncitations")
    problems += check_cites(tex)
    print("\nstyle")
    problems += check_style(tex)
    print("\nnumbers against committed artifacts")
    problems += check_numbers(tex)

    print()
    if problems:
        for p in problems:
            print("FAIL: %s" % p)
        return 1
    print("WRITEUP OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
