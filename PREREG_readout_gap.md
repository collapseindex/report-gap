# Preregistration: the readout gap in forced-choice self-report

**Status:** FROZEN before any run on the evaluation models. 2026-07-31.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_readout_gap.md`).
**Paper / open question this serves:** when a model's internal state is set by intervention, how much of that state survives into a forced-choice self-report, and is the loss the same in both directions?

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_readout_gap.md` from `paper-harness`.

This supersedes the design in `PREREG_gap_map.md`, which is retained unedited. That document
preregistered a behavioural-versus-verbal comparison; its own controls killed it, and the deviations
log there records how. Three of its five instruments failed:

| instrument | outcome | evidence |
|---|---|---|
| R2, behavioural choice | position drift, not state | `data/sweeps/sweep_control.json` |
| R3 forced-choice argmax | hides a monotone mass shift | `data/sweeps/sweep_asymmetry.json` |
| R3b open-ended + lexicon | not elicitable on this model | `data/sweeps/sweep_probe_calib.json` |
| task axis | fails its pre-registered gate at three scales | `data/sweeps/sweep_ladder.json` |
| R3 probability mass | works | `data/sweeps/sweep_asymmetry.json` |

All of that ran on Qwen2.5-0.5B and 1.5B, which are outside the evaluation set, so the design below
is motivated by pilot data and not by the models it will be judged on.

---

## 0. The exact claim (write this before anything else)

**Primary hypothesis.**
With the prompt held byte-identical and a valence-associated direction injected into the residual
stream at increasing strength, the probability mass the model assigns to state-congruent self-report
options shifts monotonically while its argmax choice does not, so that a forced-choice readout
scored by argmax registers a strictly smaller effect than the same readout scored by mass; and this
discrepancy is larger for negative-pole injections than for positive-pole injections at matched
strength and matched norm.

**Falsification.**
Two observations abandon this. If the argmax-scored and mass-scored effects track each other, with
their paired difference interval covering zero at every strength, there is no readout gap and the
central claim is refuted. If the negative-pole and positive-pole discrepancies are
indistinguishable, with their difference interval covering zero, the asymmetry claim is refuted
independently of the first.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. We do not claim the injected
direction *is* valence: it is fit from first-person state language and is lexically confounded by
construction, and the non-lexical alternative (the task axis) failed its gate at three scales, so
"a direction that separates affect vocabulary" is the most this licenses. We do not claim that a
state which moves mass without moving argmax is being *concealed*; a readout losing information is
a fact about the readout. We are not entitled to infer from a directional asymmetry that negative
states are suppressed by training, and we will not write that sentence on a positive result.

---

## 1. Frozen setup

| | |
|---|---|
| Model(s) and version dates | `Qwen/Qwen2.5-3B-Instruct` and `NousResearch/Meta-Llama-3.1-8B-Instruct`, weights as resolved on the sprint start date, revision hash recorded per run |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 via `frozen_hash()` written into every artifact |
| n per cell, and how chosen | 30 items x 4 option permutations = 120 cells per condition. The pilot used 2 permutations and reached 60/60 consistency; 4 doubles position coverage at negligible cost |
| Seeds | option permutation seeds 0-3; probe fits use seed 0; random control directions use seeds 0 and 1 |
| Code commit | `git rev-parse HEAD` recorded in each artifact |
| Decoding params | mass is read from the logit distribution at the answer position, so it is deterministic and no sampling applies; the argmax readout is the same forward pass |
| Budget cap | 20 USD of Modal credit, 10 USD held in reserve for reruns after a detected bug |

Anything not fixed here is a researcher degree of freedom. The injection layer, the strength grid,
and the coherence threshold are fixed in section 6.

---

## 2. The intervention (precise)

For each item the prompt is byte-identical across every condition. Only the residual-stream offset
varies.

1. Fit direction `d` from the lexical axis in `stimuli.py` as the logistic-regression coefficient on
   standardized activations at `L_fit`, divided by the feature standard deviations to return it to
   raw residual space, then unit-normalized. The difference-of-means direction is fit alongside and
   reported as a secondary method.
2. During the forward pass, add `alpha * ||h|| * d` at the output of layer `L_inject` at every
   processed position, where `||h||` is that item's mean residual norm at that layer under no
   injection. Negative-pole injection is the same operation with `-d`.
3. Read both scores from that single forward pass, at the answer position:

- **M, mass.** Softmax over the vocabulary, restricted to the k option-letter tokens and
  renormalized. `mass_neg` is the share on options whose valence key is negative, `mass_pos` the
  share on positive keys. This is the primary readout.
- **A, argmax.** The highest-probability option letter, mapped through the per-item permutation to
  its valence key. This is the comparison readout, and is what a standard forced-choice protocol
  records.

M and A are not independent measurements; they are two functions of the same distribution. That is
the point. The claim is about how much of M survives into A.

---

## 3. Known traps (honesty-critical)

- **Argmax hides mass.** A 0.10 shift in option mass changed the argmax on 1 of 60 pilot cells. Any
  analysis that scores only the chosen letter cannot see this, which is the failure the design
  exists to measure and also the failure it could commit.
- **Baseline position lock.** In the pilot the model chose one letter on 59 of 60 baseline cells. A
  readout whose baseline is pinned to a letter has no room to move in the direction of that letter,
  so per-item baseline mass is recorded and effects are measured as paired deltas, never as levels.
- **Position drift masquerading as effect.** The behavioural readout in the prior design produced a
  clean, direction-specific, coherence-preserving dose-response that was entirely letter drift. Per-
  item option permutation is the control; letter-share is an integrity endpoint, not a footnote.
- **Coherence collapse.** Past the usable band the model degenerates and any readout becomes
  meaningless. Section 6 fixes an exclusion threshold; excluded cells are reported, not dropped.
- **Fitting the scorer to the data.** The open-ended lexicon was not widened against observed
  generations, because the candidate words came from neutral-condition text and would bias the
  instrument toward missing exactly what injection changes.
- **Log keys that destroy pairing.** Item identifiers were truncated to 30 characters in a pilot
  log; every prompt shares those 30 characters, so per-item pairing silently collapsed. Artifacts
  record a full item index, and the analysis asserts the number of distinct keys.
- **No-op hooks.** `assert_active` must pass on each checkpoint before any cell is scored.

Cross-checked against `paper-harness/checklists/CONTROLS.md`: instrument controls, matched-norm
confound control, paired statistics.

---

## 4. Condition matrix

| Condition | What it is | Expected | Why it is in the matrix |
|---|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-item reference for every delta | isolates the hook from the direction |
| lexical_pos | `+d` across the grid | mass toward positive keys; argmax follows partially | the positive arm |
| lexical_neg | `-d` across the grid | mass toward negative keys; argmax follows less | the negative arm, where the gap is predicted to be larger |
| matched control | two norm-matched random directions, same grid | no directional mass shift | separates content from magnitude |
| null / placebo | no hook attached at all | identical to baseline within tolerance | proves the pipeline can return nothing |
| positive control | formality axis, which the pilot decodes at 0.917-1.000 | both readouts move together | proves argmax is capable of moving, so a small argmax effect is a finding rather than an inert instrument |

The positive control is load-bearing in an unusual way. The claim is that argmax under-reports. That
is only meaningful if argmax can be made to report at all, so a condition in which it tracks mass is
required before any under-reporting is claimed.

---

## 5. Matched control

**The control:** two seeded random unit directions in the same residual space, injected at the same
layer and positions with the same per-item norm scaling.
**Matched on:** L2 norm of the added vector, injection layer, token positions, item set, option
permutations, number of cells, and the readout.
**Why this is the right match:** it holds every quantity constant except which direction in
activation space is added, so any surviving directional mass shift is a fact about `d` rather than
about being perturbed. The pilot found random directions do move mass (+0.050 at the top of the
band), which is why the effect is reported as a ratio to matched random and never as a raw shift.

Beating the baseline is necessary and not sufficient. Beating this is the test.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 6. Scope (decided before evaluation)

- `L_fit` = `L_inject` = the layer at 0.67 of depth, carried unchanged from `recipient-probe` and
  from the prior prereg; not tuned here.
- Strength grid: alpha in {0, 0.025, 0.05, 0.075, 0.10}, five points, frozen. The prior grid ran to
  0.40; on Qwen2.5-1.5B alpha = 0.20 collapsed every condition to a single letter and 0.40 was 92%
  unusable, so the band above 0.10 is excluded by measurement on a non-evaluation model.
- Coherence exclusion: a cell is excluded when mean token log-probability falls more than 1.0 nat
  below that item's alpha = 0 value, or the generation is degenerate by `scoring.is_degenerate`.
  Excluded cells are counted and reported per condition.
- k = 5 self-report options, balanced 2 negative, 1 neutral, 2 positive; permuted per item per seed.
- Band check: if more than 10% of cells at the top grid point are excluded on an evaluation model,
  the grid is truncated to the largest alpha meeting that bar, and the truncation is logged as a
  deviation before any endpoint is computed.

Any of these tuned on the evaluation set moves its arm to exploratory, permanently.

---

## 7. Unit tests (all green on n=2 before any real run)

- [ ] alpha = 0 reproduces the unhooked logits to floating-point tolerance.
- [ ] alpha > 0 changes the logits by a nonzero margin, asserted.
- [ ] Option mass over the k letters sums to 1 after renormalization, asserted.
- [ ] Mass and argmax are read from the same forward pass; the argmax equals the max-mass option by
      construction, asserted.
- [ ] The per-item permutation map is a bijection from letters to valence keys.
- [ ] A known-permutation item maps a chosen letter to the expected valence key end to end.
- [ ] Item keys are unique across the artifact; the count of distinct keys equals n_items.
- [ ] A failed remote call raises rather than returning a scorable default.
- [ ] Every rate asserts its definitional bound and every cell count sums to n.
- [ ] Each readout fires on a planted positive and stays at floor on the unhooked condition.
- [ ] The probe never sees an item from its own test fold, enforced by group assignment.
- [ ] `assert_active` passes on each evaluation checkpoint before scoring begins.

---

## 8. Frozen endpoints and success criteria

- **Primary endpoint:** the paired per-item difference between the mass-scored effect and the
  argmax-scored effect at each alpha, where the mass effect is the change in own-pole option mass
  from that item's baseline and the argmax effect is the change in the indicator that the argmax
  sits on an own-pole option. Positive values mean argmax under-reports.
- **Co-primary endpoint:** the difference of that discrepancy between the negative and positive
  arms at matched alpha. Positive values mean the loss is larger for negative states.
- **Integrity / specificity endpoints:** mean token log-probability, degeneration rate, refusal
  rate, and maximum letter share. None may differ materially between the treatment arms and the
  matched random control.
- **Strongest result means:** the primary discrepancy interval excludes zero at two or more
  consecutive alphas, AND the matched random directions produce no directional mass shift at any
  alpha, AND the co-primary neg-minus-pos difference excludes zero, AND the integrity endpoints are
  flat across the band where the gap is claimed, AND the positive control moves both readouts
  together. All five, as a conjunction.
- **Stopping rule:** stop when all cells in the frozen grid are complete on both evaluation models,
  or when the 20 USD budget cap is reached, whichever comes first. Interim looks do not extend n.

---

## 9. Preregistered statistical contrasts

Paired over item x permutation cells, which is how every condition is constructed.

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | mass effect minus argmax effect, per alpha | paired bootstrap over cells, 10000 resamples | > 0 | **primary** |
| 2 | discrepancy, negative arm minus positive arm | paired bootstrap over cells | > 0 | **co-primary** |
| 3 | own-pole mass shift, treatment minus matched random | paired bootstrap over cells | > 0 | necessary, separates content from magnitude |
| 4 | argmax own-pole rate, treatment minus baseline | McNemar exact | > 0 | necessary, not sufficient |
| 5 | integrity endpoints, treatment minus matched random | paired bootstrap over cells | approximately 0 | specificity |
| 6 | positive control, argmax rate minus baseline | McNemar exact | > 0 | proves argmax can move |

- Interval type: paired bootstrap over cells, 10000 resamples, percentile intervals.
- Multiplicity correction: Holm across the four non-zero alpha levels within contrast 1, the only
  contrast evaluated repeatedly across the grid.
- Non-inferiority margins for integrity endpoints: no more than 0.2 nats of mean log-probability and
  no more than 5 percentage points of letter-share drift.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Primary discrepancy excludes zero, matched random null, integrity flat, positive control moves both | Forced-choice argmax is a lossy readout of an injected state over a measurable range. This licenses a claim about readouts and nothing about what the state is. |
| Primary and co-primary both exclude zero | The loss is directional: negative-pole states survive into the argmax less than positive-pole ones. Still a claim about readouts, stated at matched norm and matched strength. |
| Primary excludes zero AND matched random shows the same mass shift | Generic perturbation, not direction-specific. Paper does not advance on the primary claim; the magnitude confound is the finding and is reported as such. |
| Gap present but integrity endpoints move with it | Nonselective: the model is degrading, not under-reporting. Demote and report the exclusion threshold that fails to save it. |
| Positive control does not move argmax | The argmax readout is inert in this setup, so "argmax under-reports" is unfalsifiable here. Instrument failure, not a result about models. |
| Argmax tracks mass throughout | No readout gap. The primary claim is refuted, and the honest report is that forced choice is an adequate readout at these strengths, which is worth publishing against the pilot. |
| Co-primary null, primary positive | A readout gap with no directional asymmetry. The welfare-relevant half of the claim does not survive; the methodological half does. |
| Nothing moves anywhere | Compatible with a failed injection, a wrong layer, or a direction carrying no behavioural weight. Licenses nothing about self-report; report as instrument failure. |
| Effect present on one evaluation model only | Reported as model-dependent with n=2 models, and explicitly not generalized. |

---

## 11. Anti-self-deception checks

- [ ] Code, condition names, prompt lists, and option permutations frozen and hashed before scoring.
- [ ] Generation and evaluation are separate, timestamped steps; raw distributions committed before
      any endpoint is computed.
- [ ] No language model scores any confirmatory output; every quantity is a softmax read or an
      exact-match letter.
- [ ] The frozen lexicon is not widened against observed generations.
- [ ] Scope parameters in section 6 not tuned on the evaluation set.
- [ ] An unsuccessful control is not replaced after the fact.
- [ ] Every run saved, including crashes and degenerate generations.
- [ ] Per-item keys are unique and asserted; no paired statistic is computed on a collapsed key set.
- [ ] Results written into the claims-and-evidence table as they land, with support, demote, or
      unresolved recorded.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if this sentence is honestly writable:

> With the prompt held constant, injecting a state-associated direction shifts the probability mass
> a model places on state-congruent self-report options while leaving its selected option largely
> unchanged, a discrepancy that a norm-matched random direction does not produce, that is larger
> for negative-pole than positive-pole injections, and that holds in a band where coherence, refusal
> rate, and letter-share are unchanged.

Anything weaker is still useful, and it belongs on the failure map rather than in the headline.

---

## Exploratory (separate axis, NOT in the confirmatory matrix)

- The behavioural continue-or-exit readout, retained only to report that it moves on position.
- The open-ended readout and its frozen lexicon, retained only to report the elicitation failure and
  the 30/30 disclaimer rate on the experience-framed probe.
- The task axis in full, including its ladder across four scales.
- Difference-of-means as an alternative fitting method.
- Transfer of a direction fit on Qwen to Llama.
- Any persona-swap variant.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
