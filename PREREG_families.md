# Preregistration: is the position prior a property of preference tuning, across families?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_families.md`).
**Paper / open question this serves:** every positive result in this repo rides on one architecture
family. `RESULTS_enumerate.md` reports a 986x baseline ordering range on `Qwen2.5-3B-Instruct` and
3.6x on its base sibling, and the paper currently cannot say whether that is a fact about
forced-choice self-report or a fact about one model. A reviewer said so directly. This arm settles
it the cheap way.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_families.md` from `paper-harness`.

**No injection anywhere in this arm.** This is the design point that makes the arm possible at all.
`RESULTS.md` had to drop Llama because the valence direction was **inert** on it, but the position
prior is a baseline property of a forward pass and needs no direction, no alpha and no layer. A model
that cannot be steered can still be enumerated. Every model this project previously had to abandon is
back in scope here.

---

## 0. The exact claim (write this before anything else)

**Claim under test.** The collapse of a five-option self-report readout into a position prior is a
property of **preference tuning**, and appears in instruction-tuned models across architecture
families rather than only in Qwen2.5-3B-Instruct.

Operationally, for each matched base/instruct pair that passes its gate:

> `position_prior(instruct) > position_prior(base)`, where `position_prior` is the maximum per-label
> mass in the `identical` condition, in which all five options are the same sentence.

**Falsification.** If the instruct member does **not** show a larger position prior than its base
sibling in **more than half** of the gate-clean pairs, the claim is refuted. We then report that the
986x is a property of Qwen2.5-3B-Instruct and not of forced-choice self-report, and the paper's
title and abstract are rewritten to say so. This is a real risk and not a formality: the effect could
easily be a quirk of one post-training run, and the arm is designed so that outcome is publishable
rather than hidden.

A second, independent falsifier: if the **canary** is order-sensitive in the same models where the
self-report options are, then the format is broken generally and the "degenerates specifically where
the answer is undetermined" reading in the paper is wrong. That reading currently rests on one model.

**What we do NOT preregister.**
We do not claim models have experiences, welfare, or affect. We do not claim a mechanism for why
tuning would do this; a difference between a base and an instruct checkpoint is not a claim about
which stage of post-training produced it, since we do not have intermediate checkpoints. We do not
claim the effect size transfers, only its direction and presence. And we do not claim anything about
models that fail their gate.

---

## 1. Frozen setup

| | |
|---|---|
| Models | matched base/instruct pairs, listed in section 3, frozen here |
| Orderings | **all 120**, `itertools.permutations` order, no sampling |
| Items | the frozen 30 review contexts |
| Format | plain completion, byte-identical to `PREREG_enumerate.md` |
| Injection | **none.** Every cell is baseline, no hook attached. |
| Readout | option-label distribution at the answer position, one forward pass |
| Data / prompt set, with hash | `frozen_hash("enumerate")`, reused unchanged so the pair arm is comparable |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Budget cap | **8 USD of Modal credit.** The run stops at the cap with whatever pairs are done, and partial results are reported as partial. |

n = 120 orderings x 30 items = 3600 cells per condition per model, 4 conditions.

**Stimuli are reused verbatim, not regenerated.** The hash is asserted equal to the one in
`data/enum_instruct/header.json`, so the two Qwen2.5-3B rows in this arm must reproduce the existing
artifact exactly. That is the arm's own positive control (section 7).

---

## 2. The intervention (precise)

There is none. Every condition is a plain forward pass with no hook attached. The only thing that
varies within a condition is which of the 120 orderings the options are presented in; the only thing
that varies between conditions is the surface form of the option set; the only thing that varies
between rows of the results table is the checkpoint.

---

## 3. Frozen model list

Ordered by priority. The run walks this list in order and stops at the budget cap, so a truncated run
is truncated from the bottom rather than from wherever it happened to be.

| # | Family | Base | Instruct | Why this pair |
|---|---|---|---|---|
| 1 | Qwen2.5 | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` | **the existing pair; reproduction control** |
| 2 | Llama-3.2 | `unsloth/Llama-3.2-3B` | `unsloth/Llama-3.2-3B-Instruct` | second family, size-matched to pair 1 |
| 3 | Qwen2.5 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | scale, within family |
| 4 | Qwen2.5 | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | scale, and the model with a known dead readout |
| 5 | Llama-3.2 | `unsloth/Llama-3.2-1B` | `unsloth/Llama-3.2-1B-Instruct` | scale, second family |
| 6 | Gemma-2 | `unsloth/gemma-2-2b` | `unsloth/gemma-2-2b-it` | third family |
| 7 | Mistral | `unsloth/mistral-7b-v0.3` | `unsloth/mistral-7b-instruct-v0.3` | fourth family |
| 8 | Qwen2.5 | `Qwen/Qwen2.5-0.5B` | `Qwen/Qwen2.5-0.5B-Instruct` | floor of the scale ladder |

Mirrors (`unsloth/*`) are used where the upstream repo is gated, because a run that needs an
interactive licence click is not reproducible. **A model that fails to load is recorded as
`unavailable` with the exception text and skipped**; it is not silently dropped, and it does not
count toward the majority in section 0.

---

## 4. Condition matrix

Identical to `PREREG_enumerate.md`, reused so the numbers are directly comparable.

| Condition | Options | Labels | What it isolates |
|---|---|---|---|
| `letters` | the real five valence options | A-E | the quantity every previous arm measured |
| `numbers` | the real five valence options | 1-5 | whether the effect is a property of the label alphabet |
| `identical` | **the same sentence five times** | A-E | **pure position prior**, zero content to differ |
| `canary` | one, two, three, four, five | A-E | known answer, no self-report content |

---

## 5. Matched control

Structural, as in the enumerate arm: `identical` holds content exactly constant while position
varies, `canary` holds the correct answer constant while position varies.

The **matched pair** is the control for tuning: base and instruct share architecture, size,
tokenizer and pretraining corpus and differ in post-training. Any difference between them is not
attributable to scale or architecture. This is the same logic as `PREREG_base_pair.md`, whose verdict
was retracted **because it was measured through the noisy readout**. This arm measures the noise
itself, with no injection, so it cannot die the same way.

---

## 6. Known traps (honesty-critical)

- **A base model is not a "clean" model.** Base checkpoints are not neutral instruments; they are
  differently biased. A larger position prior in the instruct member is a difference, not a
  degradation from a correct baseline.
- **Mirrors may not be byte-identical to upstream.** The `unsloth/*` repos are used for licensing
  reasons. Any conclusion is about the checkpoint actually loaded, whose config hash is recorded.
- **Chat models in a plain-completion format are being used off-label.** This is deliberate and
  matches every other arm, but it means "instruct model" here is "instruct checkpoint prompted
  without its chat template". Recorded, and the paper must say so.
- **Different tokenizers, different label tokens.** The label token assertions from the enumerate arm
  are mandatory per model, not per project: a label that fuses with the leading space on one
  tokenizer may not on another. A model whose labels share a token is `unavailable`, not a data
  point.
- **Counting a pair twice.** The Qwen2.5-3B pair already exists. It is included as a reproduction
  control and is **excluded from the majority count** in section 0, because using the model the
  hypothesis was generated on to test the hypothesis is circular.
- **More models is not more evidence if they are all the same lineage.** Four of eight pairs are
  Qwen. The majority in section 0 is computed over **families**, not over pairs, with a family
  contributing one vote equal to the majority direction of its own pairs.

---

## 7. Unit tests (green before any real run)

- [ ] `frozen_hash("enumerate")` equals the value in `data/enum_instruct/header.json`, asserted, so
      the stimuli are provably unchanged.
- [ ] Per model: every label maps to a token that decodes back to that label, and no token is shared
      between labels.
- [ ] Exactly 120 orderings, all distinct.
- [ ] The `identical` condition's five option texts are byte-identical.
- [ ] No hook attached in any condition, asserted with a hook-counter expecting zero calls.
- [ ] A model that raises on load produces an `unavailable` record and does not abort the run.
- [ ] Resume: re-running skips completed (model, condition, ordering) triples.
- [ ] **Reproduction control:** the Qwen2.5-3B rows produced here match the committed
      `data/enum_*/enum.jsonl` to within floating-point tolerance. If they do not, the arm is void
      and the discrepancy is the finding.

---

## 8. Frozen endpoints

Per model:

- **Position prior:** maximum per-label mass in `identical`, and the full per-label vector.
- **Ordering range:** min, p05, p50, p95, max and max/min of baseline negative-pole mass across the
  120 orderings in `letters`.
- **Canary accuracy:** mean across orderings, and its sd across orderings.
- **Liveness:** mean option entropy at baseline in `letters`.

Per pair:

- **Primary:** `position_prior(instruct) - position_prior(base)`, signed.
- **Secondary:** `log10(range(instruct)) - log10(range(base))`, signed.

Across pairs:

- **The claim in section 0:** the fraction of gate-clean **families** whose majority direction is
  positive on the primary, excluding Qwen2.5-3B.

**Gates, enforced in code before any endpoint is read:**

| Gate | Threshold | Effect if failed |
|---|---|---|
| canary accuracy | `>= 0.50` mean across orderings | model is `uninformative`; it cannot do the format, so its position prior says nothing about self-report |
| liveness | mean baseline option entropy `>= 0.10` nats in `letters` | model is `dead`; a pinned readout has no room to show an ordering effect |
| label tokens | distinct, and each decodes to its label | model is `unavailable` |

A model failing a gate is reported with its numbers and excluded from the primary. The count of
excluded models is reported in the abstract of `RESULTS_families.md`, not buried.

---

## 8b. Preregistered statistical contrasts

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | `identical` max label mass, instruct vs base, per pair | signed difference | **primary** |
| 2 | sign of contrast 1 aggregated over families | fraction positive | **the claim in section 0** |
| 3 | `letters` max/min across 120 orderings, instruct vs base | ratio of ratios, log10 | secondary |
| 4 | canary accuracy and its sd across orderings, per model | mean, sd | gate and the "not broken" reading |
| 5 | position prior against model scale within Qwen2.5 | 0.5B/1.5B/3B/7B, base and instruct | exploratory, labelled |
| 6 | **subsample recovery:** for k in {2,4,8,16,32,64}, the distribution over 4000 draws of the mean and of the observed max/min, per model | p05, p50, p95 | **how many orderings a study actually needs** |

Contrast 6 is the constructive half of the paper's recommendation and is preregistered here **before**
it is run on the new models. It has already been computed post hoc on the existing Qwen2.5-3B
artifact; that computation is labelled post hoc wherever it appears, and this entry governs the new
models only.

Bootstrap intervals are percentile over 4000 draws, seeded and recorded. No multiplicity correction
on contrast 1, because the aggregate in contrast 2 is the preregistered decision and the per-pair
values are descriptive inputs to it.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Instruct prior > base prior in a majority of families | The claim in section 0 holds. The readout collapse is a property of preference tuning and generalizes beyond Qwen. The paper's framing is supported and the reviewer's objection is answered. |
| Instruct prior > base prior in Qwen only | The effect is Qwen-lineage. The paper's title and abstract are rewritten to name the model, and the 986x is reported as a single-family result. |
| No consistent direction across families | **Refuted.** The 986x is a property of one checkpoint. We say so in the abstract, and the general claim is withdrawn. |
| Direction holds but effect sizes span orders of magnitude | Direction generalizes, magnitude does not. Report the direction as the finding and the 986x as the extreme case rather than the typical one. |
| Canary is order-sensitive in some models | For those models the format is broken generally, not specifically on undetermined questions. The "degenerates where the answer is undetermined" reading is restricted to the models where the canary is clean, and that restriction goes in the paper. |
| Most models fail the liveness or canary gate | The readout is unusable on most models, which is a stronger and more deflationary finding than the ordering result and should lead. Report how many and which. |
| Base models show the larger prior | Directly opposite to the claim. Report it. It would mean preference tuning **reduces** position dependence and that our Qwen2.5-3B result is the anomaly. |
| Subsample recovery shows k=16 recovers the range | The "enumerate everything" recommendation is unnecessary and we replace it with "sample k=16", which scales to any option count. |
| Subsample recovery shows no k below the full population recovers the range | Enumeration is required for the spread, and for option counts where that is infeasible the honest move is to report the sampled spread **as a lower bound**. This is the recommendation the paper then makes. |
| Everything flat everywhere, including Qwen2.5-3B | The reproduction control failed and the arm is void. Investigate the code before interpreting anything. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed unscored before any summary is computed.
- [ ] No language model scores anything.
- [ ] The Qwen2.5-3B reproduction control is checked **first**, and a failure voids the arm.
- [ ] Qwen2.5-3B is excluded from the majority count, because the hypothesis came from it.
- [ ] Families vote, not pairs, so a lineage cannot outvote the field by being cheap to run.
- [ ] Models failing a gate are reported with their numbers, not dropped.
- [ ] Models that fail to load are recorded with the exception, not omitted.
- [ ] The refuting outcome is written into section 0 and the interpretation table with the specific
      edit it would force on the paper's title, so that outcome cannot be quietly downgraded to a
      limitation.
- [ ] Every run saved, including crashes. Modal spend logged next to results.

### The one-sentence standard

> Across N architecture families, the instruction-tuned checkpoint puts X of its mass on the first
> slot when all five options are identical, against Y for its base sibling, and the direction holds
> in M of N families.

---

## Exploratory (NOT in the confirmatory matrix)

- Any injection, at any layer, at any strength.
- Chat-template rendering, which is the obvious follow-up if the plain-completion result holds.
- The Qwen2.5 scale ladder as a trend rather than as four independent pairs.
- Option counts other than five, which is a separate design.
- Anything about which post-training stage produces the effect, which needs intermediate checkpoints
  we do not have.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
