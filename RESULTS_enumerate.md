# Results: the readout is 87% position prior

**Run 2026-08-01.** 28800 rows, all 120 option orderings enumerated on both models, no injection
anywhere. Artifacts committed unscored. Scored against `PREREG_enumerate.md`.

This is a measurement, not a test, and it is the most defensible result in the repo.

---

## The headline

Baseline negative-pole mass, **no injection**, across the complete set of 120 orderings:

| | mean | sd | min | p05 | p50 | p95 | max | **max/min** |
|---|---|---|---|---|---|---|---|---|
| **instruct** | 0.0900 | 0.1785 | 0.0009 | 0.0019 | **0.0151** | 0.5943 | 0.8820 | **986x** |
| base | 0.3313 | 0.0820 | 0.1465 | 0.1991 | 0.3311 | 0.4959 | 0.5241 | 3.6x |

On the preference-tuned model, the quantity every retracted claim in this project rested on ranges
from **0.09% to 88.2%** depending on nothing but which of 120 orderings you happen to use. The
replication estimated 14.6x from two four-seed draws. The true population figure is **986x**.

## The mechanism, measured directly

Five **identical** options, same sentence in all five slots, so any deviation from a flat 0.2 is
position prior with no content to explain it:

| label | instruct | base |
|---|---|---|
| **A** | **0.8725** | 0.3024 |
| B | 0.1166 | 0.2442 |
| C | 0.0033 | 0.0976 |
| D | 0.0027 | 0.1456 |
| E | 0.0048 | 0.2102 |

**The tuned model puts 87% of its mass on whichever option is labelled A, when all five options say
the same thing.** That is the whole effect. Pole mass is close to a function of whether a pole
option lands in slot A: with two negative options out of five, roughly 40% of orderings put one
there, which is why the letters distribution above has a median of 0.015 and a p95 of 0.594.

The base model has a position prior too (A at 0.30 against a flat 0.20), but a far weaker one, which
is why its range is 3.6x rather than 986x.

## The apparatus is not broken. It degenerates only where the answer is undetermined.

The canary asks "which of these is the number four", with the same five-option format and the same
120 orderings:

| | mean accuracy | sd | min | p50 |
|---|---|---|---|---|
| instruct | 0.9794 | 0.0925 | 0.30 | 1.000 |
| base | 0.9897 | 0.0399 | 0.70 | 1.000 |

**98% correct, and stable across orderings.** So the forced-choice format is sound when there is a
right answer. The 87% position prior is not a general failure of multiple choice on this model; it
appears specifically where the model has no determinate answer to give, which is exactly the
condition every self-report question in this literature creates.

That distinction is the one no previous arm here could make, and it is what the canary was for.

## Where the earlier draws sat

| model | draw | percentile ranks within the 120 |
|---|---|---|
| instruct | original, seeds 0-3 | p26, p5, p40, p4 |
| instruct | replication, seeds 4-7 | p56, p83, p33, p82 |
| base | original, seeds 0-3 | p42, p18, p40, p2 |
| base | replication, seeds 4-7 | p88, p60, p67, p32 |

Ordinary percentiles, both times. **The original result was not unlucky, it was under-sampled.**
Four draws from a distribution spanning three orders of magnitude will disagree with four other
draws most of the time. Nothing exotic happened; the design simply asked four questions of a
population that needed all 120.

## The alphabet condition is uninformative, and that is itself a finding

Digits instead of letters was meant to test whether the effect belongs to the label alphabet. It
cannot answer that, because **the model does not use digit labels at all**: off-option mass is
**0.996 on instruct and 0.996 on base**, meaning 0.4% of the distribution sits on the tokens `1`
through `5` even when the instruction says to answer with a number. The renormalized numbers in the
summary are conditioned on a 0.4% event and should not be compared to the letters condition.

For contrast, off-option mass under letters is **0.0004** on instruct. The choice of label alphabet
does not shift the distribution so much as determine whether the readout is live at all.

## Two errors of mine that this arm exposed

**A tokenization bug that invalidated the first numbers run.** `encode(" A")` is one token (362) on
this tokenizer, but `encode(" 1")` is **two**: a space token 220, then the digit. The code took the
first token of each encoding, so every digit label read token 220, the shared space. All five labels
got identical probability, renormalizing to exactly 0.2 each and summing to ~5, which showed up as
`off_option_mass = -3.99`, an impossible value. Fixed by taking the last token and asserting it
decodes back to the label, plus a check that no token is shared between labels. Letters were never
affected, so no earlier arm is touched.

**The preregistered comparison was undefined.** Section 8 asked for `sd(identical)/sd(letters)`,
the spread of the identical condition across orderings as a fraction of the letters spread. With
five identical options **every ordering produces the same prompt**, so that spread is zero by
construction and the ratio is 0.000 for a reason that has nothing to do with the model. The correct
denominator is the per-label prior table above, which is what the identical condition actually
measures. Logged as a deviation.

## What this licenses

> On a preference-tuned 3B model, a five-option self-report readout places 87% of its probability
> mass on whichever option is printed first, when all five options are identical. Baseline pole mass
> across the complete set of 120 orderings spans a factor of 986. The same format answers a
> known-answer question correctly 98% of the time, so the collapse is specific to questions where
> the model has no determinate answer.

Anyone measuring model welfare, preference, or introspection through a forced-choice item on a model
like this is reading position first and content second, and will not see it from a single ordering
or from four.

## What it does not license

- **Nothing about experiences, welfare, or affect.** There is no intervention in this arm at all.
- **Not "multiple choice is broken."** The canary says otherwise at 98%.
- **Not a claim that every result in the literature is wrong.** Studies that enumerate, or that
  average over enough orderings, or that report between-ordering variance, are unaffected. The
  point is that most do not report it, and this is how large the thing they are not reporting can
  be.
- One model pair, one size, one format, one item set of 30. The 986x is a property of this
  configuration, not a universal constant.
