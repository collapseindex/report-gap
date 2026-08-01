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


# Scripts that ARE the frozen protocol. They must not disclaim, because a confirmatory runner
# carrying "nothing here is a result" would be false; they must instead name the prereg they run
# under, so every script in the directory declares its status one way or the other and no script
# can be silent about which it is.
CONFIRMATORY = {"modal_readout.py", "analyze_readout.py",
                "modal_floor.py", "analyze_floor.py",
                "modal_base_pair.py", "analyze_pair.py",
                "modal_depth.py", "analyze_depth.py",
                "modal_shell_core.py", "analyze_shell_core.py",
                "modal_erase.py", "analyze_erase.py",
                "modal_enumerate.py", "analyze_enumerate.py",
                "modal_binary.py", "analyze_binary.py"}


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_declares_its_status(path):
    """Every script says whether it is exploratory or the frozen protocol. Silence is the bug.

    Exploratory scripts disclaim. Confirmatory ones name their prereg. A script that does neither
    is one whose output has no declared standing, which is how a pilot number ends up cited as a
    result months later.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    doc = ast.get_docstring(tree) or ""

    if path.name in CONFIRMATORY:
        named = re.search(r"PREREG_\w+\.md", doc)
        assert named, ("%s is listed as confirmatory but does not name the preregistration it "
                       "runs under" % path.name)
        disclaimed = re.search(r"not a (confirmatory |paper )?result"
                               r"|nothing (here )?is a (confirmatory |paper )?result", doc, re.I)
        assert not disclaimed, ("%s is the confirmatory arm but disclaims being a result; one of "
                                "the two is wrong" % path.name)
        return

    disclaimed = re.search(
        r"not a (confirmatory |paper )?result"
        r"|nothing (here )?is a (confirmatory |paper )?result"
        r"|not a measurement",
        doc, re.I,
    )
    assert disclaimed, (
        "%s does not state that its output is not a confirmatory result" % path.name
    )


def test_the_confirmatory_list_is_not_stale():
    """A name in CONFIRMATORY that no longer exists would silently exempt nothing, or worse,
    exempt a file someone later creates with that name."""
    names = {p.name for p in EXPERIMENTS}
    missing = CONFIRMATORY - names
    assert not missing, "CONFIRMATORY names scripts that do not exist: %s" % sorted(missing)


# The band-selection script may run on the EVALUATION models, which is only defensible because it
# reads headroom and never the discrepancy the paper reports. That is a claim about the code, so
# it gets asserted rather than promised.
def test_band_selection_never_computes_the_endpoint():
    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "modal_alpha_recal.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {"discrepancy_deltas", "paired_bootstrap", "mcnemar_exact", "holm",
                 "expected_discrepancy", "plant", "plant_arm"}
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    leaked = sorted(forbidden & used)
    assert not leaked, (
        "modal_alpha_recal.py runs on evaluation models and touches endpoint machinery (%s). "
        "Selecting a scope parameter with the statistic the paper reports IS tuning on the "
        "evaluation set." % ", ".join(leaked)
    )
