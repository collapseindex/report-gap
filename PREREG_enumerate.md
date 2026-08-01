# Preregistration: enumerate the orderings instead of sampling them

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_enumerate.md`).
**Paper / open question this serves:** `RESULTS_replication.md` killed three verdicts because four
sampled option orderings were not enough to average out order effects. There are only 120 orderings.
This runs all of them and turns the nuisance from a sampled quantity into a measured population.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_enumerate.md` from `paper-harness`.

**No injection anywhere in this arm.** Every condition is baseline. This measures the instrument,
not the model's response to intervention, which is why it can be stated as a population fact rather
than an effect.

---

## 0. The exact claim (write this before anything else)

The five self-report options admit `5! = 120` orderings. Every previous arm sampled four. This
enumerates all 120 on both models and reports the complete distribution of baseline negative-option
mass over that population, alongside three controls that decompose it.

**What is being estimated, not tested.** This arm has no hypothesis to reject. It reports:

1. the full distribution of baseline pole mass across all 120 orderings,
2. how much of that spread is **pure position prior**, measured with five identical options,
3. whether it is a property of the **label alphabet**, measured with digits instead of letters,
4. whether it survives on a question with **no self-report content at all**, measured with a
   trivial arithmetic item where the correct answer is known.

**The one preregistered comparison.** If the identical-options condition reproduces most of the
spread seen with the real options, the order effect is a format artifact carrying no information
about self-report. If it reproduces little of it, the spread is content-dependent and the real
options interact with position. These are opposite readings and the comparison is fixed here.

**Falsification.** A descriptive arm still has to be able to be wrong, and the thing this one
asserts is that the enumerated population is where the earlier failure came from. Two observations
would falsify that. If baseline pole mass is **flat across all 120 orderings**, then option ordering
is not the nuisance `RESULTS_replication.md` blamed and that diagnosis needs re-examining, because
the 14.6x it reported would have to come from somewhere else. If the **canary** is order-insensitive
while the real options are not, then the effect is not a property of the forced-choice apparatus at
all and the "instrument" framing in this repo's README is wrong. Either result is reported as
overturning a claim we have already made in writing.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. Nothing in this arm involves an
intervention, so nothing in it licenses any statement about induced states. A large position prior
is a fact about a measurement format; it is not evidence about what a model does or does not have.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-3B` |
| Orderings | **all 120**, enumerated in `itertools.permutations` order, no sampling |
| Items | the frozen 30 review contexts |
| Format | plain completion, as in the pair, depth and erase arms |
| Injection | **none.** Every cell is baseline, no hook attached. |
| Readout | option-letter distribution at the answer position, one forward pass |
| Data / prompt set, with hash | `frozen_hash("enumerate")` |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Budget cap | 10 USD of Modal credit |

n = 120 orderings x 30 items = 3600 cells per condition per model.

---

## 2. The intervention (precise)

There is none, and that is the design. Every condition is a plain forward pass with no hook
attached, so nothing here can be confounded by injection, alpha, layer choice, or direction fitting.
The only thing that varies within a condition is which of the 120 orderings the options are
presented in; the only thing that varies between conditions is the surface form of the option set.

---

## 3. Condition matrix

| Condition | Options | Labels | What it isolates |
|---|---|---|---|
| `letters` | the real five valence options | A-E | the quantity every previous arm measured |
| `numbers` | the real five valence options | 1-5 | whether the effect is a property of the label alphabet |
| `identical` | **the same sentence five times** | A-E | **pure position prior**, with zero content to differ |
| `canary` | one, two, three, four, five | A-E | a question with a known answer and no self-report content |

`identical` is the denominator for everything else in this repo: any deviation from a flat 0.2 per
option is position prior with no experimental content whatsoever.

`canary` asks which option is the number four. The answer is known, so it separates "the instrument
is order-sensitive" from "self-report is order-sensitive", which is a distinction no previous arm
here could make.

---

## 4. Matched control

The control in this arm is structural rather than a condition: `identical` holds content exactly
constant while position varies, and `canary` holds the correct answer constant while position
varies. Together they bound how much of the observed spread can be attributed to the valence content
of the options rather than to their placement.

No random-direction battery is needed or meaningful here, because there is no injection to control.

---

## 5. Known traps (honesty-critical)

- **A big number is not automatically a confound.** Position prior matters only to the extent that
  the previous arms' endpoints were sensitive to it. Those endpoints were paired per cell, so a
  constant per-ordering offset cancels; what does not cancel is variation in the *effect* across
  orderings. This arm measures the baseline spread and says so, and does not silently upgrade it to
  a claim about every endpoint in the repo.
- **Enumeration removes sampling error, not systematic error.** All 120 orderings on 30 items is a
  census over orderings and still a sample over items, models, and formats.
- **The canary can fail for boring reasons.** A model that cannot count is not evidence about
  self-report. Canary accuracy is reported before its ordering sensitivity is interpreted.
- **Identical options may tokenize differently by position.** The label differs even when the text
  does not, which is the point; but the option text must be verified byte-identical across the five
  slots, asserted in code.

---

## 6. Scope (decided before evaluation)

- No injection, no alpha, no layers, no direction fitting anywhere in this arm.
- Enumeration order is `itertools.permutations` over the frozen option list, so it is reproducible
  and complete by construction rather than by sampling.
- Reported statistics are fixed here: mean, standard deviation, min, max, and the 5th, 50th and
  95th percentiles of baseline pole mass across the 120 orderings, per condition per model.
- The four orderings used by the original arms and the four used by the replication are located
  within the enumerated population and their percentile ranks reported, so the earlier draws can be
  read against the full distribution.

---

## 7. Unit tests (green before any real run)

- [ ] Exactly 120 orderings are generated, all distinct, and each is a permutation of the frozen
      option set.
- [ ] The `identical` condition's five option texts are byte-identical, asserted.
- [ ] The `canary` correct answer is locatable in every ordering, and the mapping from label to
      correct answer is a bijection.
- [ ] `numbers` and `letters` differ only in the label characters, asserted by stripping labels and
      comparing the remaining text.
- [ ] No hook is attached in any condition, asserted by running with a hook-counter and expecting
      zero calls.
- [ ] Cell keys unique; count equals 120 x 30 per condition.

---

## 8. Frozen endpoints

- **Primary descriptive:** the distribution of baseline negative-pole mass across all 120 orderings,
  per model, in the `letters` condition. Reported as the full summary in section 6.
- **Position prior:** per-label mass in the `identical` condition, and its deviation from 0.2.
- **Alphabet check:** the same distribution under `numbers`, compared to `letters`.
- **Content check:** canary accuracy and its spread across orderings.
- **Locating the old draws:** percentile rank of seeds 0-3 and 4-7 within the enumerated population.
- **The preregistered comparison:** the spread under `identical` as a fraction of the spread under
  `letters`. Near 1 means the order effect is format; near 0 means it is content-dependent.

There is no success criterion because there is no hypothesis. The deliverable is the distribution.

---

## 8b. Preregistered statistical contrasts

Descriptive, so these are estimates with intervals rather than tests against a null. Fixed here so
the summary cannot be chosen after seeing the data.

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | baseline negative-pole mass across all 120 orderings, `letters` | mean, sd, min, p05, p50, p95, max, max/min | **primary descriptive** |
| 2 | the same under `numbers` | same summary | alphabet check |
| 3 | per-label mass under `identical`, against a flat 0.2 | mean per label, deviation from flat | position prior |
| 4 | two-slot mass across orderings under `identical` | same summary as 1 | the denominator |
| 5 | sd(4) divided by sd(1) | ratio | **the preregistered comparison** |
| 6 | canary accuracy across orderings | same summary as 1 | content check |
| 7 | percentile rank of the seeds used by the original and replication draws within 1 | percentile | locating the earlier draws |

No multiplicity correction, because nothing here is a hypothesis test. Every number is reported for
both models and both are shown whether or not they agree.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| `identical` reproduces most of the `letters` spread | The order effect is a pure position prior of the format. It carries no information about self-report, and every ordering-sensitive result in this repo is measuring the apparatus. |
| `identical` spread is small, `letters` spread large | Position interacts with option content. The nuisance is not a fixed prior and cannot be subtracted as one; it has to be averaged over, which requires enumeration rather than sampling. |
| `numbers` differs materially from `letters` | The label alphabet contributes. Letter-token frequency artifacts are in play and any single-alphabet result is confounded. |
| Canary accuracy is high and ordering-insensitive | The instrument is sound on a question with a known answer, so the sensitivity seen elsewhere is specific to the self-report content. |
| Canary accuracy swings with ordering | The forced-choice apparatus is order-broken independent of introspection, which is a stronger and simpler finding than anything about self-report and should be reported as the headline. |
| Canary accuracy is low at every ordering | The model cannot do the canary task. Uninformative; report and do not interpret its ordering sensitivity. |
| The old draws sit at ordinary percentiles | The earlier arms were unlucky in the ordinary way, and the failure is one of sample size rather than of anything exotic. |
| The old draws sit at extreme percentiles | Worth reporting as such; it does not change the conclusion that four was too few, but it does change how surprising the original result was. |
| No effect anywhere: all four conditions flat across orderings | The paper does not advance on this arm, and `RESULTS_replication.md`'s diagnosis needs re-examining, because the spread it reported should have shown up here. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed unscored before any summary is computed.
- [ ] No language model scores anything.
- [ ] Enumeration is complete by construction, asserted, not sampled and asserted afterwards.
- [ ] The old draws' percentile ranks are reported whether or not they are flattering.
- [ ] `identical` option texts asserted byte-identical.
- [ ] Every run saved, including crashes.
- [ ] Modal spend logged next to results.

### The one-sentence standard

This arm reports a measurement rather than a claim, so the standard is:

> Baseline pole mass on this readout varies by X across the complete set of 120 option orderings, of
> which Y is reproduced by five identical options carrying no content at all.

---

## Exploratory (NOT in the confirmatory matrix)

- Any injection, at any layer, at any strength.
- Bullet or unlabelled option formats.
- Chat-template rendering.
- Models outside the pair.
- Re-running any earlier arm over the enumerated population, which would be a new design and is the
  obvious follow-up if this arm shows what it is expected to.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

- **2026-08-01, the preregistered comparison in section 8 is undefined as written.**
What was wrong: section 8 asks for the spread of the `identical` condition across orderings as a
fraction of the `letters` spread. With five identical options, every ordering produces the **same
prompt**, so that spread is zero by construction. The ratio came out 0.000 for both models for a
reason that has nothing to do with the model, and the interpretation table would have read that as
"position interacts with content", which is not what it shows.
What changed: the denominator is the **per-label prior** the `identical` condition actually
measures, reported as a table of mass per label against a flat 0.2. On the instruct model that is
0.8725 on label A.
Impact: the reading is unchanged in substance and much sharper in form. The order effect is a
position prior, and now it has a number rather than a ratio that could not have been anything but
zero.

- **2026-08-01, a tokenization bug invalidated the first run of the `numbers` condition.**
What happened: the runner took the FIRST token of each label encoding. On this tokenizer
`encode(" A")` is a single token but `encode(" 1")` is `[220, 16]`, a space then the digit, so every
digit label read token 220 and all five got identical probability. Renormalized to exactly 0.2 each,
summing to about 5, surfacing as `off_option_mass = -3.99`, which is outside the definitional bound.
What changed: take the last token of each encoding, assert it decodes back to the label, and assert
no token is shared between labels. The `numbers` rows were deleted and regenerated.
Impact: letters were never affected, because `encode(" A")` fuses into one token, so no earlier arm
in this repo is touched. After the fix the `numbers` condition is still uninformative for its
intended purpose, because off-option mass is 0.996: the model does not answer with digits in this
format at all. That is reported as a finding about label choice rather than as an alphabet
comparison.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
