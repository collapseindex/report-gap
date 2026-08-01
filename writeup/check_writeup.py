"""Structural checks on main.tex that do not require a LaTeX toolchain.

Every check here corresponds to a way this paper could ship broken without anyone noticing at
draft stage: a \\ref to a label that does not exist renders as "??", a \\cite to a missing bib key
renders as "[?]", and a number in the prose that disagrees with the repository is the failure this
whole project is about.

    python check_writeup.py [main.tex]

The optional path exists so that `tests/test_writeup_checks.py` can point the checker at a
deliberately broken COPY and require it to fail, without ever mutating the real paper.
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
    """The repo bans em dashes and double hyphens used as punctuation.

    Grepping for U+2014 alone is not enough and was not enough: LaTeX renders `---` as an em dash,
    so 21 of them shipped in the claims tables while this check printed "none". Ranges written
    `0$--$3` or `\\ref{a}--\\ref{b}` are en dashes for ranges, not punctuation, and are allowed.
    """
    problems = []
    for i, line in enumerate(tex.splitlines(), 1):
        if "\u2014" in line or "\u2013" in line:
            problems.append("line %d: literal em/en dash character" % i)
        if re.search(r"(?<!-)---(?!-)", line):
            problems.append("line %d: LaTeX em dash (---), which renders as one" % i)
        if re.search(r"[a-zA-Z] -- [a-zA-Z]", line):
            problems.append("line %d: -- used as punctuation between words" % i)
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

    # The erase arm's numbers are re-derived through the ARM'S OWN analyzer, not reimplemented
    # here. A checker that recomputes beside the tool can agree with the paper while both are
    # wrong; one that calls the tool cannot.
    erase = ROOT / "data" / "erase_instruct" / "erase.jsonl"
    if erase.exists():
        sys.path.insert(0, str(ROOT / "experiments"))
        import analyze_erase as AE

        rows = AE.load(erase)
        n = len({r["cell"] for r in rows})
        checks.append((None, n, "erase arm distinct cells (informational)"))

        sd = statistics.pstdev([r["probe_orth"] for r in rows
                                if r["condition"] == "baseline"]) or 1.0

        def probe(r):
            return r["probe_orth"]

        no_erase = abs(AE.paired(rows, -1, "neg", "baseline", probe, sd).point)
        art, prim = {}, {}
        for L in sorted({r["erase_layer"] for r in rows if r["erase_layer"] > 0}):
            art[L] = abs(AE.paired(rows, L, "erase_only", "baseline", probe, sd).point)
            prim[L] = abs(AE.vs_random(rows, L, "neg_erase", probe, sd).point)

        # Claimed values are TYPED OUT from what the paper says, not read back from the artifact,
        # so this check can fail. A check that derives the claim from the data it is checking is
        # decorating, not validating.
        checks.append(("0.9909", round(no_erase, 4), "erase: un-erased reference"))
        checks.append(("0.0408", round(art[30], 4), "erase: artifact at L30"))
        checks.append(("0.1474", round(art[25], 4), "erase: artifact at L25"))
        checks.append(("6.1", round(prim[26] / art[26], 1), "erase: primary/artifact at L26"))
        checks.append(("20.9", round(prim[30] / art[30], 1), "erase: primary/artifact at L30"))
        checks.append((("$86\\%$", 86.0), round(100 * prim[30] / no_erase),
                       "erase: survival at L30, percent"))

    for claimed, actual, what in checks:
        if claimed is None:
            print("  %-46s actual %s" % (what, actual))
            continue
        # a claim is either a literal that both prints and parses, or (needle, value) when the
        # printed form ("$86\%$") is not the parseable one
        needle, value = claimed if isinstance(claimed, tuple) else (claimed, float(claimed))
        present = needle in tex
        agrees = abs(value - actual) < 5e-4
        print("  %-46s claims %s, artifact %.4f, in tex: %s, agrees: %s"
              % (what, needle, actual, present, agrees))
        if present and not agrees:
            problems.append("%s: tex says %s, artifact says %.4f" % (what, needle, actual))
        if not present:
            problems.append("%s: the paper no longer states %s. Update the check or the prose."
                            % (what, needle))
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = pathlib.Path(argv[0]) if argv else HERE / "main.tex"
    tex = path.read_text(encoding="utf-8")
    print("checking %s (%d lines)" % (path.name, len(tex.splitlines())))
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
