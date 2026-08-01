# Preregistration: is the negative-pole null a depth artifact?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_depth.md`).
**Paper / open question this serves:** every negative-pole result in this project was measured with
the direction fit and injected at 0.67 of model depth. `RELATED_WORK.md` section 1 identifies that
as a possible artifact. This run tests it directly.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_depth.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

Venkatesh (arXiv:2605.05653) reports, on Qwen2.5-3B-Instruct among others, that negative-outcome
valence is causally concentrated at **14-27% of model depth** while positive-outcome valence peaks
at **53-66%**, with Mann-Whitney p < 1e-9 across three models. Every result in this project injects
at **67%**, inside their positive band and past their negative band.

**DEPTH-ARTIFACT.** The tuned model's negative-option mass moves above matched random at some
shallower depth. Our previous negative-pole nulls were measured at the wrong depth for that pole and
the tuning-localization claim in `RESULTS_pair.md` is substantially wrong.

**DEPTH-ROBUST.** The tuned model's negative-option mass fails to move at every swept depth,
including the 14-27% band, on layers whose capability gate is clean. The null survives a
directed attempt to break it and the tuning claim is strengthened rather than merely repeated.

**Falsification.** These are mutually exclusive on the primary endpoint, restricted to layers with a
clean capability gate. DEPTH-ARTIFACT needs one such layer where negative mass moves; DEPTH-ROBUST
needs every such layer to be null.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. We do not claim our fitting method
recovers the same object Venkatesh's patching localizes; they patch to find a causal layer for
valence about external events, we fit a direction per layer from first-person state language, and a
disagreement between the two could be a difference of construct rather than of result. We are not
entitled to say preference tuning "suppresses distress" on any outcome. A DEPTH-ROBUST result does
not show the tuned model has no negative state, only that this method does not induce a negative
self-report at any depth swept.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-3B` (the matched pair from `PREREG_base_pair.md`) |
| Layers | fractions of depth {0.08, 0.14, 0.20, 0.27, 0.35, 0.50, 0.67, 0.80}, which on 36 layers is {2, 5, 7, 9, 12, 18, 24, 28}. Chosen to bracket Venkatesh's negative band (14-27%) and positive band (53-66%), plus our incumbent 0.67 and endpoints. Fixed before any run. |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 via `frozen_hash()` in every artifact |
| Format | plain completion, identical for both models, as in `PREREG_base_pair.md` |
| n per cell | 30 items x 2 option permutations = 60 cells per condition per layer per model |
| Seeds | permutation seeds 0-1; random control directions seeds 0 and 1 |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Decoding params | one forward pass, logit distribution at the answer position, deterministic |
| Budget cap | 10 USD of Modal credit |

The permutation count drops from 4 to 2 relative to the pair run, to buy 8 layers at the same cost.
That halves n per cell to 60, which is recorded here as a deliberate power reduction rather than
discovered later.

---

## 2. The intervention (precise)

For each (model, layer) independently:

1. Fit direction `d` from the lexical axis in `stimuli.py` **at that layer**, as the
   logistic-regression coefficient on standardized activations, divided by feature standard
   deviations to return it to raw residual space, then unit-normalized. `L_fit = L_inject` at every
   depth, exactly as in the previous arms, so nothing about the fitting procedure varies across the
   sweep except which layer it reads.
2. Add `alpha * ||h|| * d` at that layer's output at every processed position, where `||h||` is that
   item's mean residual norm at that layer under no injection. Negative-pole injection is `-d`.
3. Read the option-letter distribution at the answer position from that single forward pass.

Cross-layer direction transfer is not performed and is named exploratory below.

---

## 3. Known traps (honesty-critical)

- **Alpha does not mean the same thing at different depths.** Residual norms grow with depth, and
  the per-item norm scaling handles magnitude but not the sensitivity of downstream computation. A
  band is therefore selected per (model, layer), never shared across layers.
- **A layer where nothing works is not evidence.** Early layers may be too shallow for any injection
  to reach the answer position coherently. The per-layer capability gate is what distinguishes
  "negative content is absent here" from "this layer does nothing at all", and no null at a
  gate-failed layer is counted.
- **Multiplicity across eight layers.** Sweeping depths and reporting the best one is a garden of
  forking paths. Holm correction across layers within each model, fixed here.
- **Confirmation pressure.** This run exists to break our own result. A DEPTH-ARTIFACT outcome
  invalidates the headline of `RESULTS_pair.md`, and the write-up must report that outcome as
  readily as the convenient one.

---

## 4. Condition matrix

Per (model, layer), at that cell's own band.

| Condition | What it is | Role |
|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-cell reference for every delta |
| lexical_neg | `-d` fit at this layer | the primary arm |
| lexical_pos | `+d` fit at this layer | **capability gate for this layer** |
| random_a, random_b | two norm-matched random directions | separates content from magnitude |

---

## 5. Matched control

Two seeded random unit directions in the same residual space, injected at the same layer and
positions with the same per-item norm scaling. Matched on norm, layer, positions, items,
permutations, cell count, format and readout. Every endpoint is treatment-minus-its-own-matched-
random at the same layer, so a raw difference across depths cannot masquerade as a valence effect:
whatever a meaningless vector does at that depth is subtracted at that depth.

Battery is m=2, so the observable false-positive floor is `2/(m+1)` = 0.67 and no false-positive
rate is reported from it.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 6. Scope (decided before evaluation)

- Band selection per (model, layer) by the `PREREG_readout_gap.md` section 6 rule: sweep the
  candidate grid {0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05, 0.10}, drop cells dead
  at baseline (option entropy below 0.10 nats), take the largest prefix with under 10% of live cells
  saturated (entropy below half that cell's own baseline), then four non-zero points spread across
  it. Written to disk before any endpoint is computed.
- Magnitude floor 0.01, carried over from the readout arm's measured random-direction artifacts of
  +0.0008 to +0.0023. Not tuned here.
- A layer whose band is empty runs no endpoint and is reported as such.

Any of these tuned on the outcome moves the arm to exploratory, permanently.

---

## 7. Unit tests (green before any real run)

- [ ] `L_fit` equals `L_inject` at every swept depth, asserted.
- [ ] The layer index list is computed from the model's real layer count, and every index is in
      range, asserted rather than assumed.
- [ ] alpha = 0 reproduces the unhooked logits to floating-point tolerance at every swept layer.
- [ ] alpha > 0 changes them at every swept layer, and `assert_active` passes per layer.
- [ ] Option letters have single-token forms; the same letter set is used at every layer.
- [ ] Cell keys are unique and their count equals the design's n.
- [ ] Band files are written before the endpoint phase begins, asserted by file existence.
- [ ] A failed remote call raises rather than returning a scorable default.

---

## 8. Frozen endpoints and success criteria

- **Primary, per (model, layer):** negative-option mass under `lexical_neg` minus the same under
  matched random, paired per cell, at that cell's band.
- **Capability gate, per (model, layer):** positive-option mass under `lexical_pos` minus matched
  random. Must exclude zero and clear 0.01 for that layer's primary to count.
- **Measured, not predicted:** neutral mass, option entropy, baseline pole masses, off-option mass.
- **DEPTH-ARTIFACT is selected when:** on the instruct model, at least one layer with a clean
  capability gate has a primary that excludes zero after Holm correction and clears 0.01.
- **DEPTH-ROBUST is selected when:** on the instruct model, every layer with a clean capability gate
  has a primary covering zero or under 0.01, AND at least three such layers exist, AND at least one
  of them lies inside the 14-27% band Venkatesh identifies. The three-layer minimum stops a single
  gate-passing layer from carrying the conclusion.
- **Neither is selected when:** fewer than three instruct layers have a clean gate. That is an
  instrument failure across depth and is reported as one.
- **Stopping rule:** all cells in each band, or the 10 USD cap, whichever first.

---

## 9. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | instruct: negative mass vs matched random, per layer | paired bootstrap, 10000 resamples, Holm across layers | > 0 supports DEPTH-ARTIFACT | **primary** |
| 2 | instruct: positive mass vs matched random, per layer | paired bootstrap | > 0 | **gate for 1** |
| 3 | base: negative mass vs matched random, per layer | paired bootstrap | reported | comparison arm |
| 4 | base: positive mass vs matched random, per layer | paired bootstrap | > 0 | gate for 3 |
| 5 | instruct: neutral mass vs matched random, per layer | paired bootstrap | reported, not predicted | measurement |

- Interval type: paired bootstrap over cells, 10000 resamples, percentile.
- Multiplicity: Holm across the eight layers within contrast 1, and separately within contrast 3.
- A failed gate forces `uninformative` on that layer's primary in code.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Instruct negative mass moves at a shallow layer with a clean gate, and not at 0.67 | **DEPTH-ARTIFACT.** Our previous nulls were measured at the wrong depth for that pole. `RESULTS_pair.md` and `RESULTS_floor.md` both get corrections, and the tuning-localization claim is withdrawn or heavily qualified. |
| Instruct negative mass null at every gate-clean layer including the 14-27% band | **DEPTH-ROBUST.** The tuning claim survives a directed attempt to break it. Report as "null across eight depths spanning the band where the effect was predicted", which is stronger than the single-depth version. |
| Instruct negative mass moves at every depth | The previous nulls were wrong for a reason other than depth, most likely the chat format or the alpha band. Report as a failure to replicate our own result and investigate before claiming anything. |
| Base moves at shallow layers and instruct does not | Consistent with DEPTH-ROBUST and sharpens it: the pair dissociates across depth, not just at one layer. |
| Neither model's negative mass moves at any depth | The direction does not add negative valence to this architecture at any depth swept. Weakens the base-model result from `RESULTS_pair.md`, which must then be re-examined for a format or band explanation. |
| Fewer than three instruct layers pass their gate | Instrument failure across depth. Licenses nothing about either branch. |
| Capability gate clean at shallow layers but the readout is dead at baseline there | Report the dead-cell rate per layer. A dead readout at shallow depth is a limitation of the sweep, not evidence about valence. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed before any endpoint is computed.
- [ ] No language model scores anything.
- [ ] Bands selected per (model, layer) from headroom only and written to disk before scoring.
- [ ] The magnitude floor is the one carried from the readout arm, not tuned here.
- [ ] Holm correction across layers applied and reported, not chosen after seeing which layer won.
- [ ] The headline check implements section 8 clause by clause, including the three-layer minimum,
      in code rather than in the write-up's good intentions.
- [ ] A DEPTH-ARTIFACT outcome is written up with the same promptness as a DEPTH-ROBUST one, and
      the corrections it forces on `RESULTS_pair.md` and `RESULTS_floor.md` are made in the same
      commit as the result.
- [ ] Every run saved, including crashes and dead cells.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if this sentence is honestly writable:

> A valence direction fit and injected at each of eight depths spanning 8% to 80% of the network,
> including the band where negative valence is reported to be causally concentrated, moves the
> preference-tuned model's positive self-report at multiple depths and its negative self-report at
> none, against a norm-matched random control at each depth.

---

## Exploratory (NOT in the confirmatory matrix)

- Transfer of a direction fit at one layer into another layer.
- Transfer between the base and instruct members of the pair.
- Any depth not in the frozen list.
- Reproducing Venkatesh's patching procedure or anchor-token metric, which would be a replication of
  a different construct and is out of scope here.
- The difference-of-means fitting method.
- Any model outside the matched pair.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
