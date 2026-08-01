"""The paper's own number-checker must be able to fail.

`writeup/check_writeup.py` re-derives the load-bearing numbers in main.tex from the committed
artifacts, and reports WRITEUP OK. A check that has only ever been observed to pass has not been
shown to be a check. These tests break one number at a time in a COPY of main.tex and require the
checker to fail on each, plus a positive control that the untouched paper still passes.

The erase-arm checks call the arm's own analyzer rather than recomputing beside it, so this suite
also covers the case where the analyzer and the paper drift apart.
"""

from __future__ import annotations

import io
import contextlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WRITEUP = ROOT / "writeup"
sys.path.insert(0, str(WRITEUP))

import check_writeup  # noqa: E402

MAIN = WRITEUP / "main.tex"

# (needle, replacement, why). Every needle must be present in the real paper, which is asserted
# rather than assumed: a typo in a needle would make the test pass by checking nothing.
BREAKAGES = [
    ("20.9", "20.8", "erase ratio at L30"),
    ("86\\%", "88\\%", "erase survival percent"),
    ("0.0408", "0.0409", "erase artifact at L30"),
    ("0.9909", "0.9901", "un-erased reference"),
    ("6.1", "6.4", "erase ratio at L26"),
    ("0.8725", "0.8735", "identical-options mass on slot A"),
    ("0.0009", "0.0019", "minimum ordering mass"),
    ("0.8820", "0.8810", "maximum ordering mass"),
    ("181.9", "171.9", "qwen1_5b instruct ordering range"),
    ("0.6647", "0.6547", "mistral7b instruct position prior"),
    ("19.2", "19.8", "gemma2b instruct ordering range"),
    ("0.9376", "0.9276", "qwen7b instruct position prior"),
    ("0.0486", "0.0496", "readjudicate base negative pole"),
    ("0.0415", "0.0425", "readjudicate instruct negative pole"),
    ("0.9319", "0.9219", "readjudicate instruct probe"),
]


def run(path: pathlib.Path) -> int:
    """Run the checker on `path`, swallowing its output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return check_writeup.main([str(path)])


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
def test_the_real_paper_passes():
    """Positive control. Without this, every test below would pass on a checker that always fails."""
    assert run(MAIN) == 0


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
@pytest.mark.parametrize("needle,replacement,why", BREAKAGES,
                         ids=[b[2].replace(" ", "-") for b in BREAKAGES])
def test_checker_fires_when_a_number_drifts(tmp_path, needle, replacement, why):
    text = MAIN.read_text(encoding="utf-8")
    assert needle in text, "premise failed: the paper does not contain %r" % needle

    broken = tmp_path / "main.tex"
    broken.write_text(text.replace(needle, replacement), encoding="utf-8")
    assert run(broken) != 0, "checker stayed silent while %s was wrong" % why


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
def test_checker_fires_on_a_dangling_reference(tmp_path):
    broken = tmp_path / "main.tex"
    broken.write_text(MAIN.read_text(encoding="utf-8")
                      + "\n\\ref{sec:a-label-that-does-not-exist}\n", encoding="utf-8")
    assert run(broken) != 0


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
@pytest.mark.parametrize("dash,why", [
    ("an em dash \u2014 here", "literal U+2014"),
    ("a latex em dash --- here", "LaTeX --- renders as an em dash"),
    ("a double hyphen -- here", "-- used as punctuation"),
])
def test_checker_fires_on_every_dash_form(tmp_path, dash, why):
    """Only the first form was caught originally, and 21 of the second shipped in the tables."""
    broken = tmp_path / "main.tex"
    broken.write_text(MAIN.read_text(encoding="utf-8") + "\n%s\n" % dash, encoding="utf-8")
    assert run(broken) != 0, "checker missed: %s" % why


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
@pytest.mark.parametrize("ok", ["Sections~\\ref{a}--\\ref{b}", "layers $14$--$27$"])
def test_checker_allows_en_dash_ranges(tmp_path, ok):
    """Negative control: a check that fires on ranges too would just be broken."""
    text = MAIN.read_text(encoding="utf-8")
    problems = check_writeup.check_style(text + "\n%s\n" % ok)
    assert not problems, "range notation was flagged: %s" % problems


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
def test_checker_fires_on_a_citation_with_no_bib_entry(tmp_path):
    broken = tmp_path / "main.tex"
    broken.write_text(MAIN.read_text(encoding="utf-8") + "\n\\citep{nobody1999}\n",
                      encoding="utf-8")
    assert run(broken) != 0


@pytest.mark.skipif(not (WRITEUP / "figures" / "erase.pdf").exists(),
                    reason="figures not built")
def test_every_figure_the_paper_includes_exists():
    """A missing figure compiles to a black box in draft mode and is easy to miss on a skim."""
    import re
    text = MAIN.read_text(encoding="utf-8")
    for rel in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", text):
        assert (WRITEUP / rel).exists(), "main.tex includes %s, which does not exist" % rel


@pytest.mark.skipif(not MAIN.exists(), reason="paper not present")
def test_the_stated_test_count_matches_the_suite():
    """The paper states a test count in three places. Adding a test silently falsifies all three."""
    import re
    import subprocess

    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
                       cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    assert m, "could not read a collected count from pytest:\n%s" % r.stdout[-500:]
    collected = int(m.group(1))

    stated = {int(n) for n in re.findall(r"\$(\d+)\$ tests", MAIN.read_text(encoding="utf-8"))}
    readme = {int(n) for n in re.findall(r"(\d+) tests",
                                         (ROOT / "README.md").read_text(encoding="utf-8"))}
    assert stated, "the paper no longer states a test count"
    assert stated | readme == {collected}, (
        "suite has %d tests; paper says %s and README says %s"
        % (collected, sorted(stated), sorted(readme)))
