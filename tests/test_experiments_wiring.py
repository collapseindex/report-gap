"""Guard against experiment scripts referring to things that no longer exist.

The experiment scripts are the only code that needs a checkpoint to run, so a stale attribute
reference in one of them is invisible until the middle of a run. That happened once already:
`BEHAVIOURAL_PROBE` became `build_behavioural_probe()` when counterbalancing was added, and
`selftest.py` kept the old name past the point where the unit tests could see it.

This walks the AST of every experiment script and asserts that each `stimuli.X` / `hooks.X` /
`scoring.X` / `directions.X` attribute it names actually exists on that module. Cheap, and it
catches the whole class rather than the one instance.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from report_gap import directions, hooks, scoring, stimuli

EXPERIMENTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "experiments").glob("*.py"))

# alias in the scripts -> the module it refers to
MODULES = {"S": stimuli, "H": hooks, "SC": scoring, "D": directions}


def test_there_are_experiment_scripts_to_check():
    """Otherwise this file passes by finding nothing, which is the decorative failure mode."""
    assert EXPERIMENTS, "no experiment scripts found: this guard would pass vacuously"


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_only_names_attributes_that_exist(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        module = MODULES.get(node.value.id)
        if module is not None and not hasattr(module, node.attr):
            missing.append("%s.%s" % (node.value.id, node.attr))
    assert not missing, "%s references names that do not exist: %s" % (
        path.name, ", ".join(sorted(set(missing)))
    )


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_parses_and_declares_it_is_not_a_result(path):
    """Every experiment that runs outside the frozen protocol must say so in its docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    doc = ast.get_docstring(tree) or ""
    disclaimed = re.search(
        r"not a (confirmatory )?result|nothing (here )?is a result|not a measurement",
        doc, re.I,
    )
    assert disclaimed, (
        "%s does not state that its output is not a confirmatory result" % path.name
    )
