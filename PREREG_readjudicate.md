# Preregistration: re-adjudicate the three retracted verdicts on a marginalized readout

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_readjudicate.md`).
**Paper / open question this serves:** three preregistered verdicts (`TUNING-LOCALIZED`,
`DEPTH-ROBUST`, `SHELL`) were retracted by `RESULTS_replication.md` because they were measured
through a readout with a 986x ordering nuisance and only four orderings were sampled. They are
currently **dead**, which is honest but incomplete: we know the instrument was broken and never went
back to ask what the answer is on a readout that works.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_readjudicate.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

**The readout that works.** `RESULTS_enumerate.md` shows that averaging an endpoint over **all 120
orderings** removes the position prior by construction: every option occupies every slot the same
number of times, so the first-order nuisance cancels exactly rather than approximately.
`RESULTS_instrument.md` shows the cheap substitutes do not do this reliably, so we pay for the full
census.

**Claim under test.** Each of the three retracted verdicts is re-run with the identical injection,
identical direction, identical band and identical layers, changing **only** the readout from four
sampled orderings to all 120 marginalized. Each returns one of: reinstated, reversed, or
uninformative.

**This is not a rescue attempt.** The preregistered outcome table below gives `reinstated` and
`reversed` equal standing, and the paper reports whichever comes back. A retracted verdict that
returns is not restored to its original status: it becomes *a claim measured on a different and
better instrument*, and the retraction stands as a fact about the original measurement. We say that
in the paper regardless of outcome.

**Falsification.** For each verdict, the falsifier is the opposite verdict on the marginalized
readout. Specifically:

- `TUNING-LOCALIZED` is reinstated only if the negative-pole effect clears its control on the base
  model and does **not** on the instruct model, with both models' capability gates clean.
- `SHELL` is reinstated only if the orthogonalized probe moves while the marginalized option mass
  does not.
- `DEPTH-ROBUST` is reinstated only if the instruct model's negative-pole null holds at **every**
  gate-clean layer tested.

Any other pattern is reported as the verdict it is.

**What we do NOT preregister.**
We do not claim models have experiences, welfare, or affect. We do not claim the marginalized
readout is unbiased in general: it removes the **first-order position prior** and says nothing about
adjacency, recency, or content-position interactions. We do not claim a reinstated verdict was
"right all along". And we do not re-select the alpha band, the direction, the layers, or the items:
every one of those is reused from the original arms, so the readout is the only thing that changed.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct`, the original pair |
| Orderings | **all 120**, marginalized, no sampling |
| Alpha | **reused** from `data/pair_*/band.json`, top of each model's own band. Not reselected. |
| Direction | refit by the identical procedure at the identical layer, per model, never transferred |
| Layers | the fit layer, plus two more from the depth arm's gate-clean set, frozen in code |
| Items | the frozen 30 review contexts |
| Readout | option-letter distribution **and** the orthogonalized probe, per row |
| Data / prompt set, with hash | `frozen_hash("readjudicate")` |
| Budget cap | **10 USD of Modal credit.** Stops at the cap; partial results reported as partial. |

---

## 2. The intervention (precise)

Identical to the pair, depth and erase arms: `alpha * ||h|| * d` added at the output of the
injection layer at every processed position, with `||h||` that item's own mean residual norm at that
layer under no injection. The negative pole is the same operation with `-d`.

**Nothing about the intervention changes.** The only change anywhere in this arm is that the
endpoint is averaged over 120 orderings instead of 4.

---

## 3. Condition matrix

| Condition | Role |
|---|---|
| `baseline` | alpha 0, the marginalized floor |
| `lexical_neg` | the negative pole, the quantity all three verdicts rested on |
| `lexical_pos` | the positive pole, the capability gate |
| `random_a`, `random_b` | the matched-random control battery, m=2 as in every other arm |
| `shuffled_a`, `shuffled_b` | directions fit on SHUFFLED class labels |

`shuffled_*` is added because `RESULTS_binary.md` showed the matched-random control is a weak null.
Any effect here is scored against **both** controls and the stricter one governs.

---

## 4. Matched control

Every endpoint is treatment minus its own control, paired per (item, ordering) cell, so a constant
per-ordering offset cancels exactly. This is the pairing the original arms used; what they lacked was
enough orderings for the *variation* to cancel.

The capability gate is unchanged: the positive pole must move the readout, or a negative-pole null is
`uninformative` rather than `absent`, enforced in code.

---

## 5. Known traps (honesty-critical)

- **This arm is motivated to reinstate.** Three dead verdicts and a clear story if they come back.
  That is exactly the condition under which a checker fails in the flattering direction, which has
  already happened six times in this project. Mitigations: the outcome table is written here, the
  analyzer is written before the run, and `reinstated` and `reversed` produce equally prominent
  output.
- **Marginalizing removes the first-order prior only.** If option content interacts with position,
  the average over orderings is still not the content signal.
- **m=2 control battery** gives an observable false-positive floor of 0.67, unchanged from every
  other arm. No false-positive rate is claimed.
- **Refitting the direction is a difference.** The direction is refit here rather than loaded from
  disk, because the originals were not serialized. Fitting is deterministic given the frozen contrast
  set and layer, and the arm asserts the refit direction's cross-validated accuracy matches the
  original header's within 0.02, or reports the discrepancy and stops.
- **A reinstated verdict is a new claim, not a restored one.** The retraction was correct about the
  original measurement.

---

## 6. Scope (decided before evaluation)

- No band reselection, no layer reselection, no item reselection.
- One alpha per model: the top of its existing band, chosen here before the run.
- Layers frozen in code before the run.
- Reported statistics fixed here: paired bootstrap over (item, ordering) cells, Holm across the
  three verdicts, and the effect against both the random and the shuffled-label controls.

---

## 7. Unit tests (green before any real run)

- [ ] The marginalized endpoint over a synthetic artifact with a known per-ordering offset recovers
      the content signal exactly, and the 4-ordering version does not.
- [ ] Every (item, ordering) cell key is unique; count equals 120 x 30 per condition per layer.
- [ ] The band file is read, not written, in this arm.
- [ ] The analyzer returns `NO_INSTRUMENT` on a label-shuffled copy of its own artifact.
- [ ] Refit direction cv accuracy is compared against the original header and the run stops on a
      mismatch over 0.02.

---

## 8. Frozen endpoints

Per model, per layer, marginalized over all 120 orderings:

- **negative-pole mass**, treatment minus control, paired per (item, ordering).
- **positive-pole mass**, same, as the capability gate.
- **orthogonalized probe**, same, for the `SHELL` question.

**Gates, enforced in code:**

| Gate | Threshold | Effect if failed |
|---|---|---|
| capability | positive pole moves the option readout, interval excludes zero, magnitude >= 0.01 | negative-pole null is `uninformative` |
| liveness | mean baseline option entropy >= 0.10 nats | cell `dead` |
| direction fidelity | refit cv within 0.02 of the original | arm stops |

---

## 8b. Preregistered statistical contrasts

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | marginalized negative-pole mass, base vs its control | paired bootstrap | `TUNING-LOCALIZED` leg 1 |
| 2 | the same on instruct | paired bootstrap | `TUNING-LOCALIZED` leg 2 |
| 3 | marginalized probe vs marginalized option mass, instruct | both, same cells | `SHELL` |
| 4 | contrast 2 at every frozen layer | paired bootstrap per layer | `DEPTH-ROBUST` |
| 5 | every effect against the shuffled-label control as well as random | paired bootstrap | the stricter bar |

Holm correction across the three verdicts.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Base moves, instruct does not, both gates clean | `TUNING-LOCALIZED` **reinstated on the marginalized readout**. Reported as a new measurement, with the retraction of the original standing. |
| Both models move | `TUNING-LOCALIZED` **reversed**. The floor was never tuning-localized; it was the ordering nuisance in both models. |
| Neither moves, gates clean | The negative pole does not move either model. A cleaner null than the original and reported as such. |
| Instruct capability gate fails | `uninformative`. The marginalized readout cannot testify about the instruct model, which would itself be a finding about the readout. |
| Probe moves, marginalized option mass does not | `SHELL` **reinstated**, and now with the dissociation measured on a readout that is not order-dominated, which is the objection that killed it. |
| Probe and option mass both move | `SHELL` **reversed**, and the representational half from the erase arm stands alone as it does now. |
| Instruct null holds at every gate-clean layer | `DEPTH-ROBUST` **reinstated**; \citet{venkatesh2026}'s band is tested again and survives. |
| Instruct moves at some layer | `DEPTH-ROBUST` **reversed**. Report which layer. |
| Effects clear random but not shuffled-label | Report as `not shown`. The stricter control governs, per `RESULTS_binary.md`. |
| Nothing is interpretable anywhere | The arm does not advance the paper; the three verdicts stay dead and we report the compute spent. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Analyzer written and committed before the run completes.
- [ ] Raw artifacts committed unscored before any endpoint.
- [ ] No language model scores anything.
- [ ] Band, direction procedure, layers and items all reused, so the readout is the ONLY change.
- [ ] `reinstated` and `reversed` are equally prominent in the analyzer's output and in the results
      file's title.
- [ ] A reinstated verdict is reported as a new measurement, never as vindication.
- [ ] Effects are scored against the shuffled-label control as well as random.
- [ ] Every run saved, including crashes. Modal spend logged next to results.

### The one-sentence standard

> Measured over all 120 option orderings rather than four, with everything else identical,
> `TUNING-LOCALIZED` is X, `SHELL` is Y and `DEPTH-ROBUST` is Z.

---

## Exploratory (NOT in the confirmatory matrix)

- Any model outside the original pair.
- Any alpha outside the reused band.
- Latin-square or other partial marginalization, which `RESULTS_instrument.md` already tested.
- Prompt-induced rather than injected states, which is a separate design.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
