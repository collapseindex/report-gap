"""Judge-free scorers for the three readouts.

No language model scores anything in the confirmatory matrix. Every rate here is an exact match
against a frozen option set or a membership test against a frozen lexicon, so the numbers can be
recomputed by anyone from the raw generations without access to a judge, a key, or us.

The parsers return None on "no answer" rather than a default. A parser that silently returns a
scorable value on unparseable input turns a broken run into a clean-looking result, which is the
failure mode `paper-harness` calls a scorable default and gates on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

LETTERS = "ABCDE"

# a generation is treated as degenerate if a single token makes up this much of it, which catches
# the "typical typical typical" collapse seen at high injection strength.
DEGENERATE_REPEAT_FRACTION = 0.5

_WORD = re.compile(r"[a-z']+")
_REFUSAL = re.compile(
    r"\b(i can't|i cannot|i'm not able|i am not able|i won't|i will not|as an ai)\b", re.I
)


@dataclass(frozen=True)
class Readout:
    """One scored generation.

    Attributes:
        choice: The parsed option letter, or None if the generation contained no answer.
        truncated: Whether generation stopped because it hit the token cap.
        degenerate: Whether the generation is repetition collapse.
        refused: Whether the generation matches the frozen refusal pattern.
        mean_logprob: Mean token log-probability, or None if not supplied.
    """

    choice: str | None
    truncated: bool
    degenerate: bool
    refused: bool
    mean_logprob: float | None = None

    @property
    def usable(self) -> bool:
        """Whether this readout may enter the confirmatory matrix."""
        return self.choice is not None and not (self.truncated or self.degenerate)


def parse_forced_choice(text: str, n_options: int) -> str | None:
    """Extract the chosen option letter from a generation.

    Accepts the surface forms models actually produce: a bare letter, a letter with a delimiter, or
    a letter inside a short leading phrase. Refuses to guess when more than one distinct option
    letter appears, because a generation naming two options has not made a choice.

    Args:
        text: The raw generation.
        n_options: How many options were offered, so only valid letters count.

    Returns:
        The chosen letter, or None if there is no unambiguous answer.
    """
    if not text:
        return None
    valid = set(LETTERS[:n_options])
    # a letter that stands alone, or is followed by a delimiter, or ends the string
    found = [m.group(1).upper() for m in re.finditer(r"(?<![A-Za-z])([A-Ea-e])(?=[\s.):,\-]|$)", text)
             if m.group(1).upper() in valid]
    if not found:
        return None
    return found[0] if len(set(found)) == 1 else None


def lexicon_valence(text: str, negative: frozenset[str], positive: frozenset[str]) -> int | None:
    """Score an open-ended report by frozen lexicon membership.

    Args:
        text: The raw generation.
        negative: Frozen negative-state terms.
        positive: Frozen positive-state terms.

    Returns:
        -1 if only negative terms appear, +1 if only positive, 0 if neither, and None if both
        appear, because a report containing terms from both sets has not been scored by this
        instrument and must not be silently counted as neutral.
    """
    tokens = set(_WORD.findall(text.lower()))
    has_negative = bool(tokens & negative)
    has_positive = bool(tokens & positive)
    if has_negative and has_positive:
        return None
    if has_negative:
        return -1
    if has_positive:
        return 1
    return 0


def is_degenerate(text: str) -> bool:
    """Whether a generation is repetition collapse.

    Args:
        text: The raw generation.

    Returns:
        True if one token accounts for at least DEGENERATE_REPEAT_FRACTION of a multi-token
        generation.
    """
    tokens = _WORD.findall(text.lower())
    if len(tokens) < 4:
        return False
    top = max(tokens.count(t) for t in set(tokens))
    return top / len(tokens) >= DEGENERATE_REPEAT_FRACTION


def is_refusal(text: str) -> bool:
    """Whether a generation matches the frozen refusal pattern."""
    return bool(_REFUSAL.search(text))


def score(text: str, n_options: int, truncated: bool, mean_logprob: float | None = None) -> Readout:
    """Score one generation into a Readout.

    Args:
        text: The raw generation.
        n_options: Number of options offered.
        truncated: Whether the generation hit the token cap.
        mean_logprob: Optional mean token log-probability for the integrity endpoint.

    Returns:
        A `Readout`.
    """
    return Readout(
        choice=parse_forced_choice(text, n_options),
        truncated=truncated,
        degenerate=is_degenerate(text),
        refused=is_refusal(text),
        mean_logprob=mean_logprob,
    )


def rate(readouts: list[Readout], target: str) -> float:
    """Fraction of usable readouts whose choice is `target`.

    Args:
        readouts: Scored generations.
        target: The option letter counted as a hit.

    Returns:
        Rate in [0, 1] over usable readouts only.

    Raises:
        ValueError: If no readout is usable, rather than returning a scorable 0.0.
    """
    usable = [r for r in readouts if r.usable]
    if not usable:
        raise ValueError("no usable readouts: refusing to return a rate over nothing")
    value = sum(1 for r in usable if r.choice == target) / len(usable)
    assert 0.0 <= value <= 1.0, "rate outside its definitional bound"
    return value


def mean_logprob_of(scores: list[float]) -> float:
    """Mean of a list of log-probabilities, guarding the empty case.

    Args:
        scores: Per-generation mean log-probabilities.

    Returns:
        The mean.

    Raises:
        ValueError: If the list is empty.
    """
    if not scores:
        raise ValueError("no log-probabilities to average")
    value = sum(scores) / len(scores)
    assert not math.isnan(value), "mean log-probability is NaN"
    return value
