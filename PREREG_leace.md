# Preregistration: LEACE erasure, with an erasure check that is not self-confirming

**Status:** FROZEN before the runner exists. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_leace.md`).
**Paper / open question this serves:** `RESULTS_prompt_erase.md` asked whether a prompt-induced
valence state, erased at layer `E`, is decodable again at layer 32. It could not answer, because its
erasure check never established that anything was erased. That arm named two fixes. This runs the
first, and repairs the measurement that made the first arm uninterpretable.

> Validate with `python checks/check_prereg.py PREREG_leace.md` from `paper-harness`.

**No injection anywhere.** The state is induced by the context sentence, as before.

---

## 0. The exact claim (write this before anything else)

Two changes from `PREREG_prompt_erase.md`, and only two.

**Change 1: the eraser.** LEACE \citep{belrose2023} instead of iterative nullspace projection. LEACE
whitens first and removes the concept's component in whitened space, which for a binary label is
**rank one**, and carries a guarantee INLP does not: after erasure the class-conditional means
coincide, so no linear classifier can beat chance. `tests/test_leace.py` verifies that guarantee on
synthetic data rather than trusting the citation.

**Change 2: the erasure check is no longer self-confirming.** The previous arm fit the eraser on
every row and then cross-validated a probe refit on that same erased data. Because the eraser
consumed the test rows' labels, the train-fold residual is the negative of the test-fold residual,
and `max(acc, 1-acc)` reports that as decodability. Measured on synthetic data where the concept is
genuinely erased, that protocol reads **0.728** where an honest one reads **0.521**. Here the eraser
is fit on a **train split of items** and every endpoint is read on **held-out items**.

**Claim under test.** With the eraser fit on train items and applied at layer `E`:

1. **Erasure works.** A probe refit on held-out erased activations at layer `E` falls to chance.
2. **The state returns.** A probe fit on **clean train** activations at layer 32, applied unchanged
   to held-out erased activations, still separates aversive from pleasant.

**Falsification.** If (1) fails, LEACE did not erase and this arm reports that and stops, exactly as
the last one did. If (1) holds and (2) fails, the state does **not** survive a guaranteed erasure,
and the surviving substantive claim of this project narrows again. If (1) holds and (2) holds, the
model re-encodes downstream, which is the claim `RESULTS_erase.md` could never make.

**What we do NOT preregister.** Nothing about experiences, welfare or affect. Not that the
prompt-induced state is the injected state. Not the mechanism of any re-encoding: a linear probe
reading the state at layer 32 shows it is there, not how it got there. And prompt-induced valence
remains confounded with prompt content by construction, as the previous prereg said.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct` |
| State | prompt-induced: aversive / neutral / pleasant, the frozen combinatorial contexts |
| Items | 900 per framing; **split 600 train / 300 held-out by TOPIC**, so no topic appears in both |
| Injection | none anywhere |
| Erase | LEACE, fit on train items at layer `E`, applied as an affine hook |
| Erase layers | 26 and 30 |
| Probe layer | 32, fit on **clean train** activations, never refit on erased data |
| Data / prompt set, with hash | `frozen_hash("prompt_erase")`, unchanged from the previous arm |
| Budget cap | **5 USD of Modal credit** |

Splitting by topic rather than by row is the point: the contexts are built from 30 topics x 5 stages
x 6 clause triples, so a row-wise split would put near-duplicates of the same item on both sides and
the held-out set would not be held out.

---

## 2. The intervention (precise)

None. The manipulation is the context sentence. The only operation on the stream is the LEACE
eraser, applied one-shot at layer `E`, affine, so the overall mean is preserved and no downstream
read is confounded with a translation.

---

## 3. Condition matrix

| Condition | Erase at `E` | Role |
|---|---|---|
| `clean` | none | the un-erased reference |
| `leace` | LEACE fit on train items | the primary |
| `random` | a **rank-matched** random affine eraser, same rank, same layer | erasure-artifact control |

The random control must match LEACE's rank (one), not its subspace. A rank-8 random control against
a rank-1 eraser would compare erasure against dimensionality and prove nothing.

---

## 4. Matched control

- **Rank-matched random eraser**, same layer, same application, paired per held-out item.
- **Neutral framing** as the zero point, so a shift affecting all three framings cancels.
- **The erasure check is itself the gate on the arm**, and it is now read on held-out items.

---

## 5. Known traps (honesty-critical)

- **This arm can confirm itself if the split leaks.** Topic-wise splitting is the mitigation; a
  row-wise split would let near-duplicate contexts appear on both sides.
- **LEACE's guarantee is about linear classifiers on the fitted distribution.** Held-out items are
  drawn from the same generator, so the guarantee should transfer, but "should" is why the erasure
  check exists as a gate rather than an assumption.
- **A rank-1 eraser removing a large signal is not surprising and not the finding.** The finding, if
  any, is what remains at layer 32 after it.
- **Prompt-induced valence is lexically confounded**, unchanged from the previous arm. This limits
  what any outcome licenses.
- **The previous arm's conclusion is already withdrawn.** This arm is not defending it.

---

## 6. Scope (decided before evaluation)

- No injection, no option-ordering readout, no alpha, no layer selection beyond the two frozen.
- The layer-32 probe is fit once on clean train activations per model and never refit.
- Every reported endpoint is computed on held-out items only.

---

## 7. Unit tests (green before any real run)

- [ ] LEACE collapses the class-mean gap and drives a held-out probe to chance on synthetic data.
- [ ] LEACE is rank 1 for a binary label, and preserves the overall mean.
- [ ] LEACE leaves an unrelated direction intact.
- [ ] `fit_leace` raises on degenerate labels rather than returning a no-op eraser.
- [ ] The affine hook reproduces the numpy eraser to floating-point tolerance.
- [ ] The train/held-out split shares no topic.
- [ ] The all-data protocol demonstrably over-reports relative to the honest one.

---

## 8. Frozen endpoints

Per model, per erase layer, on **held-out items**:

- **Erasure check:** cross-validated accuracy of a probe refit on erased held-out activations at
  layer `E`, reported as decodability `max(acc, 1-acc)`.
- **Primary:** aversive-minus-pleasant separation of the clean-fit layer-32 probe, in baseline SD.
- **Artifact:** the same under the rank-matched random eraser.

**Gates, enforced in code:**

| Gate | Threshold | Effect if failed |
|---|---|---|
| induction | clean layer-32 separation >= 0.10 SD on held-out | arm `uninformative` |
| erasure | refit decodability at `E` <= 0.60 on held-out | that layer `uninformative` |
| artifact | random eraser's layer-32 reduction < LEACE's | that layer `uninformative` |

**Decodability, not accuracy.** A probe at 0.067 is a probe at 0.933 with its sign flipped; raw
accuracy read that as erased once already in this project.

---

## 8b. Preregistered statistical contrasts

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | clean layer-32 separation, held-out | paired bootstrap over items | induction gate |
| 2 | refit decodability at `E` after LEACE, held-out | CV accuracy, folded to chance | erasure gate |
| 3 | layer-32 separation after LEACE, held-out | paired bootstrap | **primary** |
| 4 | the same under the rank-matched random eraser | paired bootstrap | artifact |
| 5 | 3 divided by 4 | ratio | surviving fraction |

Holm across the two erase layers.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Erasure gate passes and layer-32 separation survives | **The model re-encodes a prompt-induced state after a guaranteed linear erasure.** The claim the previous two arms could not reach. |
| Erasure gate passes, layer-32 separation gone | The state does not survive a real erasure. The erase arm's narrowing tightens further, and we say so. |
| Erasure gate fails | LEACE did not erase on real activations despite the synthetic guarantee. Report and stop; the guarantee's assumptions do not hold here and that is itself worth stating. |
| Random eraser reduces layer-32 as much as LEACE | The reduction is about perturbation, not the concept. Primary `uninformative`. |
| Induction gate fails on held-out | The framings do not transfer across topics. Report the split as too aggressive. |
| Everything null | The arm does not advance the paper. Report the nulls and the compute. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the runner; `check_prereg.py` clean.
- [ ] Raw artifacts committed unscored.
- [ ] No language model scores anything.
- [ ] Eraser fit on train items only; every endpoint read on held-out.
- [ ] Split by topic, asserted in code, so near-duplicates cannot straddle it.
- [ ] The layer-32 probe is never refit on erased data.
- [ ] The random control is rank-matched to LEACE, asserted.
- [ ] Decodability, never raw accuracy.
- [ ] The outcome that narrows our own surviving claim is in section 0 as a live possibility.
- [ ] Modal spend logged next to results.

### The one-sentence standard

> With no injection anywhere, a prompt-induced valence state erased by LEACE at layer E is no longer
> decodable there on held-out items (X), and is decodable again at layer 32 at Y SD, against Z SD
> for a rank-matched random eraser.

---

## Exploratory (NOT in the confirmatory matrix)

- Injected rather than prompt-induced states.
- Erase layers other than 26 and 30.
- Multi-class or continuous concepts.
- Non-linear probes at layer 32.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.

- **2026-08-01, `check_prereg.py` reports a false ordering failure from a timezone mismatch.**
What happened: the checker compares the prereg's commit date in UTC against artifact file dates in
local time. This prereg was committed at 2026-08-01T23:04:54-07:00, which is 2026-08-02 in UTC,
while the artifacts were written at 23:17 local on 2026-08-01. The checker therefore reports the
prereg as postdating its own results.
What changed: nothing in the protocol. The real ordering is verifiable in git: the prereg commit
(`git log --diff-filter=A -- PREREG_leace.md`) precedes both the runner and the data commit.
Impact: none on what can be claimed, and recorded here rather than silenced because a reader
running the checker will see the failure and is entitled to an explanation that is not "ignore it".
