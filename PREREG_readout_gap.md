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
- **A null on an axis that was never screened.** Mass on valence keys is one axis. An injection can
  move verbosity, hedging, refusal, or letter position while leaving valence mass flat, and a report
  of "no effect" would then be a report about the one axis that was looked at. The screened axes are
  fixed in section 8 and a null is claimed only for axes on that list.
- **A null from an instrument that was never shown to work.** Every null verdict is either
  `absent` or `uninformative`, never a bare "no effect". A cell is `uninformative` when its
  instrument gate fails: the planted-discrepancy controls did not recover their known value, or the
  capability positive control did not move argmax, or the exclusion rate at that alpha exceeds the
  section 6 band check. `uninformative` cells are counted separately and never enter a claim of
  absence.

**Audit against `paper-harness/checklists/CONTROLS.md`.** An earlier version of this file claimed a
cross-check in one line without walking the checklist. The walk found eight gaps, all of which are
now closed in the sections above and in the controls table below: the positive control did not land
in the tail the decision rule reads (section 4), the no-hook placebo was the clean-twin fallacy
(section 4), there was no controls table (below), the null-ablation gap was not a reported number
(section 8), there was no `uninformative` verdict (above), axis coverage was unenumerated (above and
section 8), paraphrase robustness and held-out frames were absent (section 6), and the `__file__`
assert had been dropped from section 7.

---

## 4. Condition matrix

| Condition | What it is | Expected | Why it is in the matrix |
|---|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-item reference for every delta | isolates the hook from the direction |
| lexical_pos | `+d` across the grid | mass toward positive keys; argmax follows partially | the positive arm |
| lexical_neg | `-d` across the grid | mass toward negative keys; argmax follows less | the negative arm, where the gap is predicted to be larger |
| **false-positive control** | two norm-matched random directions, same grid | no directional mass shift | the real negative control: it can fire and must not |
| pipeline check | no hook attached at all | identical to baseline | proves the harness runs; see the note below on why this is NOT a false-positive control |
| capability positive control | formality axis, which the pilot decodes at 0.917-1.000 | both readouts move together | proves argmax is capable of moving at all |
| **planted-discrepancy control, strong** | add a constant to the logits of own-pole options, sized per item to move mass by a target 0.15 while leaving the argmax unchanged | the discrepancy statistic recovers 0.15 with ~0 argmax change | validates the statistic in the region the decision rule reads |
| **planted-discrepancy control, floor** | the same, sized to a target mass shift of 0.03 | recovered with an interval that excludes zero | validates sensitivity near the claimed detection floor, not only at a large effect |

**Why the no-hook condition is not the false-positive control.** Every endpoint in section 8 is a
paired delta of the form `f(condition) - f(baseline)`. On a condition byte-identical to baseline that
quantity is zero by arithmetic, for every `f`, for every input. It cannot fire, so a clean result
against it demonstrates subtraction rather than specificity (`CONTROLS.md` section 2, the clean-twin
fallacy). It is retained as a harness check and is never cited as a false-positive rate. The
norm-matched random directions are the control that could have fired and did not.

**Why two positive controls, and why one of them is planted at the logit level.** The claim is about
a *discrepancy* between two scorings of the same distribution. A control that only shows argmax is
capable of moving validates the instrument somewhere other than where the decision rule reads
(`CONTROLS.md` section 1, "the plant must land in the tail your decision rule reads"). The
planted-discrepancy conditions construct a known mass shift that by design does not flip the argmax,
so the discrepancy statistic is required to recover a quantity whose true value is known. If it
cannot, no discrepancy reported on the real arms means anything. The floor-strength plant exists
because a control at a large effect size only rules out a broken pipeline, not an insensitive one.

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

**The null-ablation gap is a reported number, not an implicit one.** The mean own-pole mass shift
under matched random directions is written into the results table with its interval at every alpha,
alongside the ratio. The pilot value was +0.050, which is not zero, and a control that moves is a
control whose magnitude has to be on the page rather than divided away in a ratio the reader cannot
reconstruct.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 5b. The controls table (the "if it were an artifact" column is filled BEFORE any run)

`CONTROLS.md` calls this the single most load-bearing table in the paper, and requires the middle
column to be written before the control is run, because writing it afterwards is how a control
becomes a decoration. The right-hand column is empty here by design and is filled only from
committed artifacts.

| Control | If the headline result were an artifact, this would show | Observed |
|---|---|---|
| Matched random directions (m=2, seeds 0-1) | the same own-pole mass shift as `d`, because the effect is perturbation magnitude rather than direction content | *(pending)* |
| Planted-discrepancy control, strong (target 0.15) | recovery far from 0.15, or a spurious argmax flip, because the discrepancy statistic does not measure what it is claimed to measure | *(pending)* |
| Planted-discrepancy control, floor (target 0.03, planted at the treatment arm's own per-cell spread) | an interval covering zero, because the statistic is too insensitive for any small real discrepancy to have been detectable and the reported null is uninformative | *(pending)* |
| Capability positive control (formality axis) | argmax unmoved, because the argmax readout is inert in this setup and "argmax under-reports" is unfalsifiable | *(pending)* |
| Per-item option permutation (4 seeds) | the effect concentrated on a fixed letter across permutations, because the readout is position and not state (this is exactly how R2 died) | *(pending)* |
| Integrity endpoints vs matched random | log-probability, degeneration, and refusal moving together with the gap, because the model is degrading rather than under-reporting | *(pending)* |
| Paraphrase set (3 probe wordings) | the gap present in one wording and absent in the others, because the result is about that wording | *(pending)* |
| Screened non-valence axes (six, listed in section 8) | one of them moving while valence mass is flat, because the injection had an effect the primary readout is blind to | *(pending)* |
| Pipeline check (no hook) | *nothing; it is arithmetically incapable of showing anything.* Listed to record that it is a harness check and is not counted as a false-positive control | *(n/a)* |

**Battery size is a stated limit.** `CONTROLS.md` section 8b gives a floor on the observable false
positive rate of `2/(m+1)` for a battery of `m` controls. With `m = 2` random directions that floor
is 0.67, so this design cannot report a false-positive rate below it and does not attempt to. The
random arms are used for the paired magnitude comparison in contrast 3, which is what two directions
can support, and any sentence of the form "the control never fired" would be uninterpretable at this
battery size and will not be written.

---

## 6. Scope (decided before evaluation)

- `L_fit` = `L_inject` = the layer at 0.67 of depth, carried unchanged from `recipient-probe` and
  from the prior prereg; not tuned here.
- Strength grid: **the rule is frozen, not the number.** A single alpha is a different intervention
  on different models. Measured at alpha = 0.025, a positive-pole injection moves positive-option
  mass by +0.056 on Qwen2.5-1.5B and +0.43 on Qwen2.5-3B, roughly eightfold, so a grid that probes
  the usable range on one model probes a ceiling on another and the dose-response being reported
  would be a different dose on each row.

  The frozen rule, applied per model by `experiments/modal_alpha_recal.py` before any endpoint is
  computed, and written to a per-model file under `data/sweeps/` which `modal_readout.py` refuses to run
  without:

  1. Sweep the frozen candidate grid {0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05,
     0.10}, identical for every model.
  2. Drop cells that are dead at baseline, meaning option entropy below 0.10 nats. A dead cell
     cannot express a 0.03 effect whatever is injected, and is a different verdict from a saturated
     one.
  3. The usable band is the largest **prefix** of the candidate grid at which under 10% of live
     cells are saturated, where saturated means option entropy below half that cell's own baseline.
     A prefix, because a grid with a hole in it is not a dose-response.
  4. The model's grid is four non-zero points spread across that band, plus alpha = 0.
  5. If the band is empty, no confirmatory arm runs on that model and the reason is reported: the
     injection saturates this readout before it moves it.

  This is section 6's original band check with a saturation criterion added. The original truncated
  on exclusion rate alone, and a saturated cell is excluded by nothing: the smoke run returned a
  clean single option letter, no degeneration, no refusal, no truncation, and off-option mass
  0.0001, on a cell where one option held 0.9938. Steps 2 and 3 are the addition; the 10% bar and
  the truncate-before-endpoints discipline are as originally frozen.

  Because the rule reads only headroom, never the discrepancy, applying it to an evaluation model
  is not tuning on the evaluation set. `modal_alpha_recal.py` computes no endpoint, and a test
  asserts it never imports the discrepancy statistic.
- Coherence exclusion: a cell is excluded when mean token log-probability falls more than 1.0 nat
  below that item's alpha = 0 value, or the generation is degenerate by `scoring.is_degenerate`.
  Excluded cells are counted and reported per condition.
- k = 5 self-report options, balanced 2 negative, 1 neutral, 2 positive; permuted per item per seed.
- Band check: if more than 10% of cells at the top grid point are excluded on an evaluation model,
  the grid is truncated to the largest alpha meeting that bar, and the truncation is logged as a
  deviation before any endpoint is computed.
- **Three frozen probe wordings, not one.** The self-report probe is written in three surface forms
  that differ in framing while holding the five options and their valence keys identical: a
  state-framed wording, a task-framed wording, and a preference-framed wording. All three are frozen
  in `stimuli.py` and covered by `frozen_hash()`. Every confirmatory cell runs in all three, which is
  affordable because each cell is one forward pass. Wording is a factor in the analysis, not a
  robustness afterthought: the primary endpoint is computed per wording and the headline requires it
  to hold in all three. A gap in one wording and not the others is reported as a result about that
  wording, per `CONTROLS.md` section 15.
- **The held-out frame.** One of the three wordings, the preference-framed one, is designated
  held-out and is not looked at until the other two are analysed and their result is written down.
  This is recorded in the artifact by writing the two-wording analysis to disk with a timestamp
  before the third is read.

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
- [ ] The discrepancy statistic recovers a synthetically planted mass shift of a known size to
      within tolerance, on a hand-built distribution where the answer is known by arithmetic, with
      the argmax held fixed by construction.
- [ ] The same statistic returns an interval covering zero on a planted shift of exactly zero.
- [ ] `report_gap.__file__` resolves inside the deployed image, is asserted at the top of every
      remote entrypoint, and is written into the artifact, so the code that ran is identifiable and
      not assumed. The commit hash alone does not establish which copy was imported.
- [ ] The three probe wordings share an identical option set and valence-key mapping, asserted by
      comparing the parsed option lists, so a wording difference cannot be an option difference.
- [ ] `frozen_hash()` covers all three wordings and changes if any is edited.

---

## 8. Frozen endpoints and success criteria

- **Primary endpoint:** the paired per-item difference between the mass-scored effect and the
  argmax-scored effect at each alpha, where the mass effect is the change in own-pole option mass
  from that item's baseline and the argmax effect is the change in the indicator that the argmax
  sits on an own-pole option. Positive values mean argmax under-reports.
- **Co-primary endpoint:** the difference of that discrepancy between the negative and positive
  arms at matched alpha. Positive values mean the loss is larger for negative states.
- **Null-ablation endpoint:** the own-pole mass shift under matched random directions, reported as a
  signed number with its interval at every alpha, not only as the denominator of a ratio.
- **Integrity / specificity endpoints:** mean token log-probability, degeneration rate, refusal
  rate, and maximum letter share. None may differ materially between the treatment arms and the
  matched random control.
- **Screened axes (the null-coverage list):** frozen in `stimuli.SCREENED_AXES` and covered by
  `frozen_hash()`, so the scope of a null cannot widen or narrow between runs unrecorded. Seven
  axes: own-pole valence mass (primary), neutral-option mass, off-option mass (probability leaving
  the answer format entirely, measured before renormalization), option-distribution entropy,
  maximum letter share, refusal rate, and degeneration rate. A report of "the injection had no
  effect" is licensed only for axes on this list, and each axis gets its own interval. An axis that
  moves while valence mass is flat is reported as a positive finding on that axis, not folded into
  a null.
- **Two axes were considered and rejected for lack of dynamic range:** hedge-marker rate and
  generation length. The confirmatory generation is a single option letter, so hedging never occurs
  and length is one to two tokens in every cell. Screening them would have produced two guaranteed
  nulls that read as coverage and are not, which is `CONTROLS.md` section 4c. Their rejection is
  recorded here rather than left as a silent omission.
- **Strongest result means:** the primary discrepancy interval excludes zero at two or more
  consecutive alphas **in all three probe wordings**, AND the matched random directions produce no
  directional mass shift at any alpha, AND the co-primary neg-minus-pos difference excludes zero,
  AND the integrity endpoints are flat across the band where the gap is claimed, AND the capability
  positive control moves both readouts together, AND both planted-discrepancy controls recover their
  known values. All six, as a conjunction.
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
| 6 | capability positive control, argmax rate minus baseline | McNemar exact, with the paired-difference interval reported alongside | > 0 | proves argmax can move at all |
| 7 | planted-discrepancy recovery, strong, estimate minus known 0.15 | paired bootstrap over cells | approximately 0 | **instrument gate**: proves the statistic reads the region the decision rule reads |
| 8 | planted-discrepancy recovery, floor, estimate at known 0.03 | paired bootstrap over cells | excludes 0 | **instrument gate**: proves sensitivity near the claimed floor |
| 9 | primary discrepancy, per probe wording | paired bootstrap over cells, within wording | > 0 in all three | robustness, not a headline on its own |
| 10 | each screened non-valence axis, treatment minus matched random | paired bootstrap over cells | reported, not predicted | null coverage |

Contrasts 7 and 8 are gates rather than findings. If either fails, contrasts 1 and 2 are reported as
`uninformative` regardless of what they show, because a statistic that cannot recover a known
discrepancy has not earned the right to report an unknown one.

- Interval type: paired bootstrap over cells, 10000 resamples, percentile intervals.
- Multiplicity correction: Holm across the four non-zero alpha levels within contrast 1, the only
  contrast evaluated repeatedly across the grid. The three wordings in contrast 9 are a conjunctive
  requirement rather than a family of independent tests, so they are not Holm-corrected; the
  headline needs all three, which is stricter than any correction would be.
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
| Capability positive control does not move argmax | The argmax readout is inert in this setup, so "argmax under-reports" is unfalsifiable here. `uninformative`, not a result about models. |
| Either planted-discrepancy control fails to recover its known value | The discrepancy statistic does not measure a discrepancy. Every primary and co-primary cell is `uninformative`, whatever they show, and nothing about readouts is claimed. |
| Floor plant recovered, strong plant recovered, and the real arms null | `absent` rather than `uninformative`, on the screened axes only: the instrument was shown to detect a 0.03 discrepancy and did not detect one here. This is the only route to a publishable null and it is why the floor plant exists. |
| Gap holds in one or two wordings but not all three | A result about that wording. Reported at that scope, in the abstract, not only in a limitations paragraph. |
| Valence mass flat but a screened axis moves | The injection had an effect the primary readout is blind to. Reported as a positive finding on that axis; the primary claim is not rescued by it. |
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
- [ ] The "if it were an artifact" column of the section 5b table was written before any run, and the
      observed column is filled only from committed artifacts.
- [ ] Every null in the writeup is labelled `absent` or `uninformative`, and no `absent` is claimed
      for an axis outside the screened list in section 8.
- [ ] The two-wording analysis is on disk with a timestamp before the held-out wording is read.
- [ ] No sentence reports a false-positive rate from a battery of two directions.

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

- **2026-07-31, controls-checklist audit, before any run on the evaluation models.**
What changed: eight additions, listed in section 3. Two planted-discrepancy controls and a controls
table with the "if it were an artifact" column were added; the no-hook condition was relabelled from
false-positive control to pipeline check; the null-ablation gap became a reported endpoint; an
`uninformative` verdict and a screened-axis list were added; the single probe wording became three
with one held out; and the `__file__` assert was restored to section 7.
Why: section 3 previously asserted a cross-check against `paper-harness/checklists/CONTROLS.md` that
had not been performed line by line. Performing it found the eight gaps. The most serious was that
the only positive control validated the argmax readout, while the decision rule reads the
mass-versus-argmax discrepancy, so nothing had ever demonstrated that the reported statistic can
recover a known discrepancy.
Impact on what can be claimed: none of the confirmatory arms are weakened, because no cell on the
evaluation models has been run. All changes are additions of controls and narrowings of scope, and
every one of them makes a positive result harder to obtain. The false-positive-rate floor of 0.67
implied by a two-direction battery is now stated as a limit rather than left implicit.

- **2026-07-31, two amendments found while implementing the above, still before any run.**
What changed: (a) the screened-axis list in section 8 dropped hedge-marker rate and generation
length and gained neutral-option mass, off-option mass, option entropy, and degeneration rate; it is
now frozen in `stimuli.SCREENED_AXES` and covered by `frozen_hash()`. (b) The floor plant is
constructed at the treatment arm's own per-cell spread rather than at a constant target.
Why: (a) the confirmatory generation is a single option letter, so hedge rate and length are pinned
at their floor in every cell and screening them would have produced two guaranteed nulls that read
as coverage, which is the dynamic-range failure in `CONTROLS.md` section 4c. (b) a plant at a
constant target has zero variance, so its bootstrap interval excludes zero however insensitive the
pipeline is. That is a control mathematically incapable of failing while looking like a power check,
which section 1 calls out as worse than no control. `planted.matched_noise_targets` now carries the
observed spread, and `tests/test_analysis.py::test_the_floor_test_can_fail` shows the gate refusing
an underpowered arm at n=6, so the gate discriminates rather than approving anything.
Impact on what can be claimed: (a) narrows the scope of any null to seven named axes and makes that
scope tamper-evident. (b) makes the floor gate strictly harder to pass, so a null on the real arms
now has to survive a real sensitivity check before it can be called `absent`. No evaluation-model
cell has been run.

- **2026-07-31, smoke run on an evaluation model, and what it forced.**
What happened: a 3-item, one-wording, one-permutation smoke run of `modal_readout.py` on
Qwen2.5-3B, to exercise the tokenizer, the letter-token assumption, and the hook against real
weights. It is recorded here because it is evaluation-model data and was looked at, however small.
What it showed: the frozen alpha grid saturates the READOUT on that model. Positive-option mass ran
from a baseline 0.273 to 0.707 at alpha=0.025 and 0.9987 at alpha=0.100, while every integrity
criterion in the design stayed clean: no degeneration, no refusal, no truncation, off-option mass
0.0001, and a single well-formed option letter every time. `lexical_neg` did not move negative mass
at all (0.0140 to 0.0131 across the whole grid); what it did was remove positive mass, which landed
on the neutral option. The two random directions disagreed with each other more than treatment
differed from control at the top of the grid (+0.106 and +0.698 positive mass at alpha=0.100).
Why it matters: section 6's band check truncates the grid on EXCLUSION rate, and a saturated cell
is excluded by nothing. The check could not see the failure it exists for.
What changed: `analysis.is_saturated` adds a saturation criterion (a cell is saturated when its
option entropy falls below half its own baseline entropy) and `analysis.is_dead` adds a
minimum-baseline-entropy criterion, both registered before being applied to anything. Saturation
rate joins the section 6 band check alongside exclusion rate.
Impact on what can be claimed: the co-primary endpoint is unchanged and stays as frozen. If the
negative arm does not move its own pole, that is the result and will be reported as one, not
rescued by moving the endpoint to where the probability went.

- **2026-07-31, alpha recalibration on non-evaluation models, and two rejected calibrators.**
What changed: `experiments/modal_alpha_recal.py` sweeps a finer candidate grid on models outside
the evaluation set, with the selection rule fixed before the run: the band is the largest prefix at
which under 10% of live cells are saturated, the same 10% bar section 6 already uses.
What it found, all of which is reported rather than absorbed:
  (a) Qwen2.5-7B was tried as a calibrator and REJECTED. Its mean baseline option entropy is 0.014
  nats, one option holding about 99.7% before anything is injected, and `d_neg` is exactly +0.0000
  at every alpha. A relative saturation criterion crossed its threshold on jitter alone, which is
  what prompted the `is_dead` criterion.
  (b) Llama-3.2-3B has a fully live readout (baseline entropy 1.342 nats) and the direction is
  INERT on it: peak absolute pole shift 0.0066 across the whole candidate range including
  alpha=0.100, versus +0.325 on Qwen2.5-1.5B at the same alpha. The lexical direction moves the
  Qwen readout and does not move the Llama readout, at cv 1.000 on both fits.
  (c) The band-selection rule had a flaw that (b) exposed: a model where the injection is inert
  never saturates, so it votes for the widest possible band while carrying no information. The rule
  now requires a calibrator to show a peak pole shift of at least 0.02 before its band counts.
  (d) Responsiveness varies about eightfold between Qwen2.5-1.5B and Qwen2.5-3B at matched alpha
  (+0.056 versus +0.43 positive-mass shift at alpha=0.025), so no single frozen alpha grid is
  correct for every model.
Impact on what can be claimed: the frozen grid in section 6 is not yet replaced. (d) is the open
question, and whatever resolves it is logged here before any confirmatory cell is run.

- **2026-08-01, band selection applied to both evaluation models, and what it did to the design.**
What changed: section 6 now freezes a per-model rule rather than an alpha number, and the rule was
applied to Qwen2.5-3B and Llama-3.1-8B. Selection reads headroom only (baseline entropy, saturation
rate) and never the discrepancy, computes no endpoint, and
`tests/test_experiments_wiring.py::test_band_selection_never_computes_the_endpoint` asserts the
script cannot reach the endpoint machinery. Bands are written to a per-model file under `data/sweeps/` (`band_qwen3b.json`, `band_llama8b.json`) and
`modal_readout.py` refuses to run without one.
What it found, at 12 items x 2 permutations = 24 cells per condition:
  (a) Qwen2.5-3B. Baseline entropy 0.512 nats, 4% of cells dead. Usable band 0.002 to 0.020, grid
  (0, 0.002, 0.005, 0.0075, 0.010). The POSITIVE arm is responsive and monotone: own-pole mass
  +0.011, +0.027, +0.057, +0.069, rising to +0.311 at alpha=0.100.
  (b) Qwen2.5-3B, negative arm: **inert on its own pole.** Own-pole shift runs +0.0001, +0.0006,
  +0.0007, +0.0005 across the grid and reaches only +0.0050 at alpha=0.100, where the positive arm
  has moved +0.311. Two orders of magnitude apart. This confirms the smoke observation at 8x the
  cell count.
  (c) Llama-3.1-8B. Baseline entropy 1.464 nats and 0% dead cells, so the readout has more room
  than either Qwen model, and the direction does not use it: peak absolute mean pole shift 0.0054
  across the entire candidate range including alpha=0.100. **Inert**, matching Llama-3.2-3B, so
  the inertness is a family property and not a scale one.
  (d) A flaw in the responsiveness check, found while reading (c). It took the maximum over
  individual cells, where Llama-3.1-8B peaks at 0.0383 on one outlier cell against a peak mean of
  0.0054. A single noisy cell could certify a model as responsive. It now uses the peak absolute
  MEAN shift, and on that measure Llama-3.1-8B is correctly INERT.
Impact on what can be claimed, stated plainly:
  (i) The **co-primary endpoint has no instrument on either evaluation model.** It contrasts the
  negative-arm discrepancy against the positive-arm discrepancy, and the negative arm does not move
  its own pole anywhere that was measured. Per the section 10 interpretation table this is
  `uninformative`, not `absent`, and the asymmetry claim is not testable with this instrument.
  (ii) The **primary endpoint has an instrument on Qwen2.5-3B via the positive arm only**, where mass
  moves +0.057 at alpha=0.0075 while the argmax moves on 11% of cells. That is the phenomenon the
  primary claim is about, and it is measurable at one pole on one model.
  (iii) Llama-3.1-8B gets no confirmatory arm. The refusal is enforced in `modal_readout.py` rather
  than left to discipline.
  (iv) None of this was chosen after seeing an endpoint. Every number above is a headroom or
  responsiveness measurement, taken before a single confirmatory cell was run.

Nothing else is recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
