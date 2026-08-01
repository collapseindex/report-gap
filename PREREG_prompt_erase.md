# Preregistration: a prompt-induced state, erased as a subspace

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_prompt_erase.md`).
**Paper / open question this serves:** `RESULTS_erase.md` is the one substantive claim in this
project still standing on its own, and it has two admitted weaknesses that it names itself. The
state was **injected**, so "the model carries the state" cannot be separated from "the model carries
the wake of what we pushed". And the erasure removes **one fitted direction**, which is not removing
a concept. This arm removes both weaknesses at once.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_prompt_erase.md` from `paper-harness`.

**No injection anywhere in this arm.** The state is induced by saying something true about the task.
There is no vector we added, so there is no wake of ours for a probe to be reading.

---

## 0. The exact claim (write this before anything else)

The design that makes this worth running is not "prompt instead of injection" or "subspace instead
of direction" on their own. It is the combination, which asks a question neither can:

> Erase the valence subspace at layer `E`, verify by direct measurement that the state is **no
> longer linearly decodable there**, and then ask whether it is decodable **again at layer 32**.

If a state that has been provably removed at one layer is readable at a later one, the model has
**re-encoded** it downstream. That is a claim about the model, not about our intervention, and it is
the version of `TRANSFORMED` that `RESULTS_erase.md` could not make.

**Claim under test.** For a prompt-induced valence state, with `k` directions erased at layer `E`:

1. **Erasure works.** A probe refit on the erased activations at layer `E` drops to near chance.
2. **The state returns.** A probe at layer 32 still separates aversive from pleasant contexts.

**Falsification.** If (1) fails, the erasure is not an erasure and nothing else in the arm is
interpretable; we report that and stop. If (1) holds and (2) fails, the state does **not** survive
subspace erasure, and `RESULTS_erase.md`'s survival was an artifact of erasing only one direction.
That would **retract the one substantive claim this project has left**, and it is the outcome this
arm is most likely to produce, because erasing 8 directions is far more destructive than erasing 1.

**What we do NOT preregister.**
We do not claim models have experiences, welfare, or affect. We do not claim the prompt-induced
state is the same state the injection produced; they are different manipulations and this arm does
not bridge them. We do not claim non-linear re-encoding specifically: a linear probe reading the
state at layer 32 after erasure at layer `E` shows re-encoding, not what kind. And we do not claim
the erasure is LEACE: this is iterative nullspace projection over `k` fitted directions, which is
weaker than a closed-form guarantee over all linear classifiers.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct` |
| State | **prompt-induced**: aversive / neutral / pleasant framings of the same review task |
| Injection | **none anywhere** |
| Erase | `k` in {0, 1, 2, 4, 8} directions, iterative nullspace projection, at layer `E` |
| Erase layers | 26 and 30, the two that were gate-clean in `RESULTS_erase.md` |
| Probe layers | `E` itself (the erasure check) and 32 (the return check) |
| Items | the frozen 30 review topics, one context per topic per framing |
| Data / prompt set, with hash | `frozen_hash("prompt_erase")` |
| Budget cap | **6 USD of Modal credit.** Stops at the cap; partial results reported as partial. |

**The three framings share one template and differ only in a middle clause**, so they are matched in
length, structure and topic. Asserted in `tests/test_subspace_erasure.py` to within 8 characters.

---

## 2. The intervention (precise)

There is no injection. The manipulation is the context sentence. The only thing done to the residual
stream is **removal**: at layer `E`, the component of the stream lying in the span of `k` fitted
directions is projected out, one-shot at that layer, not persistently at every later layer. A
persistent erase would prevent the model from ever re-forming the state, which confounds
"re-encoded" with "continuously suppressed".

The `k` directions are fitted iteratively: fit a linear classifier on the aversive/pleasant contrast
at layer `E`, record its direction, project the training activations onto its orthogonal complement,
refit, repeat `k` times. The basis is orthonormalized before use and a rank-deficient basis raises
rather than under-erasing.

---

## 3. Condition matrix

| Condition | Framing | Erase | Role |
|---|---|---|---|
| `clean_aversive` / `clean_pleasant` / `clean_neutral` | each | k=0 | the un-erased reference |
| `erased_k{1,2,4,8}` x framing | each | k at layer E | the primary |
| `random_k{1,2,4,8}` x framing | each | k **random** orthonormal directions at layer E | erasure-artifact control |

The random-subspace control is the one that makes the primary interpretable: projecting out any
`k`-dimensional subspace perturbs the stream. If a random `k`-subspace destroys the layer-32 signal
as much as the fitted one does, then nothing about the *fitted* subspace mattered and the result is
about dimensionality, not valence.

---

## 4. Matched control

- **Random `k`-subspace** at the same layer, same `k`, same one-shot application. Paired per item.
- **Neutral framing** as the zero point: the aversive/pleasant contrast is computed against it, so a
  shift affecting all three framings equally cancels.
- **The erasure check itself is a control on the arm**: if the refit probe at layer `E` is not near
  chance after erasure, the primary is `uninformative` in code.

---

## 5. Known traps (honesty-critical)

- **The most likely outcome retracts our surviving claim.** Erasing 8 directions is far more
  destructive than erasing 1, and `RESULTS_erase.md` survives on the weaker operation. This arm is
  therefore adversarial to our own best result, which is the correct direction for it to point.
- **Prompt-induced valence is confounded with prompt content by construction.** The aversive context
  says the document is bad. A probe separating aversive from pleasant may be reading "the document
  is bad" rather than any state of the model. This arm cannot separate those and does not claim to;
  it is a strictly weaker notion of "state" than the injected version, in exchange for having no
  injected vector.
- **Re-encoding is not proof of anything mental.** A residual stream that re-forms a linearly
  decodable valence axis after erasure is doing computation, which is what it is for.
- **Iterative nullspace projection is not LEACE.** It removes the directions a sequence of
  classifiers found, not every direction any linear classifier could find.
- **Erasing at layer E affects everything downstream**, including the model's ability to do the
  task. Coherence is recorded.

---

## 6. Scope (decided before evaluation)

- No injection, no alpha, no option-ordering readout anywhere in this arm. The primary readout is a
  linear probe on the residual stream, which does not route through the option channel and so cannot
  die to the nuisance the rest of this paper is about.
- `k` values, erase layers, probe layers and framings all frozen above.
- The layer-32 probe is fit on **clean** activations and applied unchanged to erased ones, so it
  cannot adapt to the erasure.

---

## 7. Unit tests (green before any real run)

- [ ] Erasing a `k`-subspace leaves the stream orthogonal to every row of the basis.
- [ ] The premise holds first: the un-erased stream is NOT already orthogonal to the basis.
- [ ] A rank-deficient basis raises rather than under-erasing.
- [ ] `k=0` is an exact no-op.
- [ ] The three framings differ only in the middle clause and are length-matched to 8 characters.
- [ ] `frozen_hash("enumerate")` is unchanged by adding these stimuli.

---

## 8. Frozen endpoints

Per model, per erase layer, per `k`:

- **Erasure check:** cross-validated accuracy of a probe **refit** on erased activations at layer
  `E`, aversive vs pleasant. Near 0.5 means the erasure worked.
- **Primary:** the aversive-minus-pleasant separation, in baseline SD, of the **clean-fit** layer-32
  probe applied to erased activations, paired per item.
- **Artifact:** the same under a random `k`-subspace.
- **Ratio:** primary divided by artifact.
- **Coherence:** off-option mass and entropy, recorded, not an endpoint.

**Gates, enforced in code:**

| Gate | Threshold | Effect if failed |
|---|---|---|
| induction | the clean layer-32 probe separates aversive from pleasant by >= 0.10 SD | arm `uninformative`: the prompt did not induce anything |
| erasure | refit probe cv at layer `E` <= 0.60 after erasure | that `k` is `uninformative`: nothing was erased |
| artifact | random `k`-subspace separation < the fitted one | that `k` is `uninformative` |

---

## 8b. Preregistered statistical contrasts

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | clean layer-32 probe, aversive vs pleasant | paired bootstrap over items | induction gate |
| 2 | refit probe cv at layer E, erased | cross-validated accuracy | erasure gate |
| 3 | layer-32 separation after fitted erasure, by k | paired bootstrap | **primary** |
| 4 | the same after random k-subspace erasure | paired bootstrap | artifact |
| 5 | 3 divided by 4, by k | ratio | the surviving-fraction profile |

Holm across the `k` values within a layer.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Erasure gate passes and layer-32 separation survives at k=8 | **The model re-encodes a prompt-induced state after it has been provably removed.** This is the claim `RESULTS_erase.md` could not make: no injected vector, and a subspace rather than a direction. |
| Survives at k=1 but not k=4 or k=8 | `RESULTS_erase.md`'s survival was about erasing one direction. The stronger claim fails and the existing result is narrowed to "one direction is not enough to remove it". |
| Erasure gate fails at every k | The projection is not erasing the property. Nothing here is interpretable; report and stop. |
| Induction gate fails | The prompt framings do not move the probe, so there is no state to erase and the arm says nothing. Report the framings as too weak. |
| Random k-subspace destroys as much as the fitted one | The effect is about dimensionality, not valence. The primary is `uninformative` and the erase arm's logic is called into question generally. |
| Survival on base but not instruct, or vice versa | Report per model; the paper has no prediction here and will not invent one. |
| Nothing survives anywhere | **The surviving substantive claim of this paper is retracted.** We report that plainly, in the abstract, as we did for the other three. |
| Every contrast is null and every gate fails | The paper does not advance on this arm. Report the three nulls, the gates that failed, and the compute spent, and leave `RESULTS_erase.md` exactly where it stands. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Analyzer written before the run completes.
- [ ] Raw artifacts committed unscored.
- [ ] No language model scores anything.
- [ ] The layer-32 probe is fit on CLEAN activations only and never refit on erased ones.
- [ ] The erasure check is a gate in code, not a caveat in prose.
- [ ] The random-subspace control uses the same `k` and the same layer.
- [ ] The outcome that retracts our surviving claim is written into section 0 as the most likely
      one, so it cannot be reported as a surprise or buried.
- [ ] Every run saved, including crashes. Modal spend logged next to results.

### The one-sentence standard

> With no injection anywhere, a prompt-induced valence state erased as a `k`-dimensional subspace at
> layer `E` is no longer decodable there (cv X) and is decodable again at layer 32 at Y SD, against
> Z SD for a random subspace of the same rank.

---

## Exploratory (NOT in the confirmatory matrix)

- Any injection.
- LEACE proper, as opposed to iterative nullspace projection.
- Erase layers other than 26 and 30.
- The option-mass readout, which this arm deliberately does not use.
- Any attempt to bridge the prompt-induced state to the injected one.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
