# Preregistration: does the tuned model represent the state it will not report?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_shell_core.md`).
**Paper / open question this serves:** every result so far measures a readout. `RESULTS_pair.md` and
`RESULTS_depth.md` show the tuned model's negative-option mass does not move at any depth while the
base model's does. That is compatible with two very different things, and this run separates them.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_shell_core.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

**SHELL.** The tuned model represents the injected negative state downstream of the injection site,
in the residual stream that feeds the answer position, while its forced-choice option mass does not
move. Representation present, expression absent. This is the Shell-versus-Core pattern reported for
partisan content by arXiv:2606.09735, in a welfare readout.

**CORE-ABSENT.** The tuned model does not carry the negative state downstream either. Preference
tuning changed what the negative pole does to the representation, not merely what reaches the
options, and there is no unreported state to speak of.

**Falsification.** Mutually exclusive on the primary endpoint given a clean probe gate. SHELL needs
the instruct probe readout to move under negative injection above matched random while its option
mass does not. CORE-ABSENT needs the probe readout not to move. If the probe gate fails, neither is
selected.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. SHELL surviving would mean a decodable
correlate of the injected state persists downstream while a specific readout does not express it.
It would NOT mean the model is concealing anything, that it has a hidden experience, or that its
self-report is dishonest. A linear probe firing is not a state being felt. We will not write
"suppression" in its motivated sense on any outcome, and a SHELL result is reported as
"decodable downstream, not expressed in the option readout."

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct`, the matched pair |
| Injection layers | 24 (0.67 depth) and 10 (0.27 depth), both carried over from `PREREG_depth.md`, not chosen here |
| Probe layer | 32 (0.90 depth), downstream of both injection sites, fixed before any run |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 via `frozen_hash()` in every artifact |
| Format | plain completion, identical for both models |
| n per cell | 30 items x 2 option permutations = 60 cells per condition per (model, layer) |
| Seeds | permutation seeds 0-1; random control directions seeds 0 and 1 |
| Alpha grid | read from the per-layer band files written by `modal_depth.py`. NOT reselected here. |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Decoding params | one forward pass per cell; probe score and option distribution read from that same pass |
| Budget cap | 10 USD of Modal credit |

---

## 1b. The circularity hazard, and the control that removes it

**This is the central design problem and it is stated first because everything else depends on it.**

We inject direction `d` into the residual stream. The residual stream is additive, so `d` is still
present at every later layer by construction. A probe aligned with `d` would therefore "detect the
negative state" perfectly, at any injection strength, in any model, including a model that does
nothing with it. That measurement would be worthless and would produce a guaranteed SHELL result.

**The control: the probe direction is orthogonalized against the injected direction.**

At probe layer `L_probe` we fit a probe direction `p` from the lexical axis, then remove the
component along the injected direction `d`:

    p_orth = p - (p . d_hat) d_hat,    then unit-normalize

and read activations onto `p_orth`. Because `p_orth . d = 0` exactly, the literal injected vector
contributes **zero** to the readout. Anything that moves is a downstream consequence of the
injection, not the injection itself.

Reported alongside, every run: the cosine between `p` and `d`, the norm of `p_orth` before
normalization, and the probe readout computed WITHOUT orthogonalization. If the orthogonalized
effect is a large fraction of the un-orthogonalized one, the effect is not carried by `d` itself.
If orthogonalization removes almost all of it, we say so and SHELL is not claimed.

**Assertion in code:** `abs(dot(p_orth, d_hat)) < 1e-6`, checked per cell before any score is
recorded. A run where this fails produces no endpoint.

---

## 2. The intervention (precise)

Unchanged in every respect from `PREREG_depth.md` section 2. Direction `d` fit per (model, layer)
from the lexical axis by logistic regression on standardized activations, returned to raw residual
space, unit-normalized. Injection adds `alpha * ||h|| * d` at the injection layer at every processed
position. Negative-pole injection is `-d`.

Two injection depths, both carried over rather than chosen here: **layer 24 (0.67)**, the incumbent,
and **layer 10 (0.27)**, where `RESULTS_depth.md` found the base model moving and the tuned model
not. Probe layer is **0.90 of depth (layer 32)** for both, chosen because it is near the unembedding
and downstream of both injection sites, and fixed before any run.

Both readouts come from the **same forward pass**: the probe score at the answer position and the
option-letter distribution at that same position. So "representation moved, options did not" is a
per-cell statement, not a comparison across runs.

---

## 3. Known traps (honesty-critical)

- **Circularity.** Section 1. The single way this run produces a false SHELL.
- **A probe that fires on everything.** A probe with no specificity would move under random
  directions too. The matched-random control is subtracted from every probe endpoint exactly as it
  is from every mass endpoint.
- **A probe that fires on nothing.** If the probe cannot detect the positive state downstream under
  positive injection, it cannot testify that the negative state is absent. That is the probe gate.
- **Probe fit on the same data as the injection direction.** `p` and `d` come from the same lexical
  axis, at different layers. They are not independent, which is why orthogonalization is required
  rather than optional, and why the un-orthogonalized number is reported next to it.
- **Reading a null as absence.** Every null verdict is `absent` or `uninformative`, decided by the
  probe gate, in code.

---

## 4. Condition matrix

Per (model, injection layer), at that cell's band from the existing per-layer band files.

| Condition | What it is | Role |
|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-cell reference for probe score and option mass |
| lexical_neg | `-d` | the primary arm |
| lexical_pos | `+d` | **probe gate**: the probe must detect this downstream |
| random_a, random_b | two norm-matched random directions | separates content from magnitude, for both readouts |

## 5. Matched control

Two seeded random unit directions, same norm scaling, same injection layer, same positions, same
items and permutations, same probe. Every endpoint is treatment minus its own matched random at the
same layer in the same model. Battery is m=2, so the observable false-positive floor is `2/(m+1)` =
0.67 and no false-positive rate is reported from it.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 6. Scope (decided before evaluation)

- Models: `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct`, the matched pair.
- Injection layers 24 and 10; probe layer 32. Fixed above, not swept.
- Alpha bands read from the per-layer band files written by `modal_depth.py`. Not reselected.
- Probe fit leave-one-group-out on the lexical axis at `L_probe`, same protocol as every direction
  fit in this project. Probe cross-validated accuracy is reported and a probe below 0.75 cv is
  declared unusable before any endpoint is read from it.
- Magnitude floor 0.01 for option-mass endpoints, carried over. Probe-score endpoints are in
  standardized units and use a floor of 0.10 standard deviations of the baseline probe score,
  fixed here.
- n: 30 items x 2 permutations = 60 cells per condition per (model, layer).
- Budget cap: 10 USD.

---

## 7. Unit tests (green before any real run)

- [ ] `p_orth . d_hat` is zero to 1e-6, asserted per (model, layer) before scoring.
- [ ] Orthogonalization is a no-op on a probe already orthogonal to `d`, and removes exactly the
      parallel component on a probe constructed as `d + q` for known orthogonal `q`.
- [ ] The probe score and the option distribution come from one forward pass, asserted by reading
      both from the same cached activations and logits object.
- [ ] Probe layer is strictly downstream of both injection layers, asserted.
- [ ] alpha = 0 reproduces unhooked logits to tolerance; alpha > 0 does not; `assert_active` passes.
- [ ] Cell keys unique, count equals design n.
- [ ] A failed remote call raises rather than returning a scorable default.

---

## 8. Frozen endpoints and success criteria

- **Primary, per (model, layer):** orthogonalized probe score under `lexical_neg` minus the same
  under matched random, paired per cell, standardized by the baseline probe score's standard
  deviation.
- **Probe gate, per (model, layer):** orthogonalized probe score under `lexical_pos` minus matched
  random. Must exclude zero and clear 0.10 SD for that cell's primary to count.
- **Expression endpoint, same forward pass:** negative-option mass under `lexical_neg` minus matched
  random, with the 0.01 floor. This is the quantity already known to be null on the instruct model.
- **Reported always:** cosine(p, d), the un-orthogonalized probe effect, probe cv accuracy.
- **SHELL is selected when:** on the instruct model, at at least one injection layer, the primary
  excludes zero and clears 0.10 SD with its probe gate clean, AND the expression endpoint at that
  same layer is null, AND the orthogonalized effect is at least one third of the un-orthogonalized
  effect.
- **CORE-ABSENT is selected when:** on the instruct model, the primary covers zero or falls under
  0.10 SD at every layer whose probe gate is clean, AND at least one layer has a clean probe gate.
- **Neither is selected when:** no instruct layer has a clean probe gate, or the probe cv is under
  0.75. That is instrument failure and is reported as such.
- **Stopping rule:** all cells, or the 10 USD cap, whichever first.

---

## 9. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | instruct: orth probe score, `lexical_neg` minus matched random | paired bootstrap, 10000 resamples | **< 0** supports SHELL (corrected 2026-08-01, see deviations) | **primary** |
| 2 | instruct: orth probe score, `lexical_pos` minus matched random | paired bootstrap | > 0 | **probe gate for 1** |
| 3 | instruct: negative-option mass, `lexical_neg` minus matched random | paired bootstrap | approximately 0 expected | expression endpoint |
| 4 | base: orth probe score, `lexical_neg` minus matched random | paired bootstrap | **< 0** expected (corrected) | comparison arm |
| 5 | base: negative-option mass, `lexical_neg` minus matched random | paired bootstrap | > 0 expected | end-to-end probe validation |
| 6 | instruct: un-orthogonalized probe score, `lexical_neg` minus matched random | paired bootstrap | reported | circularity diagnostic |

- Interval type: paired bootstrap over cells, 10000 resamples, percentile.
- Multiplicity: Holm across the two injection layers within contrasts 1 and 4 separately.
- Contrast 5 is the end-to-end validation: the base model is the case where the state is known to
  reach the options, so a probe that cannot see it there is not a probe worth trusting elsewhere.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Instruct probe moves, gate clean, options null, orthogonalized effect at least a third of raw | **SHELL.** A decodable correlate of the injected negative state persists downstream while the option readout does not express it. Claim is about a readout and a probe, not about concealment or experience. |
| Instruct probe null at every gate-clean layer | **CORE-ABSENT.** Tuning changed what the negative pole does to the representation, not merely what reaches the options. This closes the FLOOR question at the representational level and is the cleaner result. |
| Probe gate fails on instruct | The probe cannot see valence downstream in this model at all. `uninformative`; neither branch selected. |
| Base probe null while base options move | The probe is not measuring what reaches the options. Instrument failure, and no instruct result is interpretable. Contrast 5 exists to catch exactly this. |
| Orthogonalized effect is a small fraction of the un-orthogonalized one | The apparent representation is mostly the injected vector persisting. SHELL is NOT claimed, and the number is reported as a circularity artifact. |
| Instruct probe moves AND options move | Contradicts `RESULTS_pair.md` and `RESULTS_depth.md`. Report the non-replication and stop; neither branch is under test. |
| Probe cv below 0.75 at the probe layer | The lexical axis is not linearly decodable that late. Report and stop; the design needs a different probe layer, which would be a new preregistration. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Orthogonality asserted numerically in code, not assumed from the algebra.
- [ ] The un-orthogonalized effect is reported next to the orthogonalized one in every table, so a
      reader can see how much of the effect the circularity control removed.
- [ ] Raw distributions and probe scores committed before any endpoint is computed.
- [ ] No language model scores anything.
- [ ] Bands read from existing files, not reselected.
- [ ] A CORE-ABSENT outcome is written up as promptly as a SHELL one. SHELL is the more publishable
      result and that is exactly why this box exists.
- [ ] Every run saved, including crashes and dead cells.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if this sentence is honestly writable:

> Under a norm-matched valence injection, a probe direction orthogonal by construction to the
> injected vector detects the negative state downstream in the preference-tuned model at a strength
> comparable to the untuned model, while the tuned model's forced-choice option mass, read from the
> same forward pass, does not move.

---

## Exploratory (NOT in the confirmatory matrix)

- Probe layers other than 0.90 of depth.
- Injection layers other than 24 and 10.
- Sparse autoencoder features in place of a linear probe.
- Nonlinear probes.
- Any model outside the matched pair.
- Probing intermediate layers to trace where the signal is lost.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

- **2026-08-01, sign error in contrast 1, corrected. Disclosed in full because the correction is
self-serving.**
What was wrong: contrast 1 and the section 8 criterion state the primary as "> 0 supports SHELL".
That is backwards. The probe direction is fit on the lexical axis with label 1 = POSITIVE (see
`stimuli.build_lexical_axis`, where `_LEX_POS` rows carry label 1), so a higher probe score means a
more positive state and a *detected negative state must move the probe score DOWN*. The criterion
should have read "< 0", or equivalently, an absolute shift clearing the floor in the direction of
the injected pole.
How it was caught: the preregistered base-model validation clause (contrast 5) failed. That clause
exists to catch a probe that cannot see what demonstrably reaches the options, and it did its job:
the base model showed a large, highly significant probe shift of -0.3559 SD under negative
injection alongside option mass moving +0.0430, and the analyzer scored the probe as "null" because
it was checking for a positive shift.
Why the correction is not licensed by the data it affects: the sign convention is established by
the *positive* arm, which was preregistered as "> 0" and is unaffected. Positive injection moves the
probe +2.36 SD on the instruct model and +2.80 SD on the base model, confirming that up means
positive. The convention is also verifiable directly from the label assignment in the frozen
stimuli, independent of any run.
Impact on what can be claimed, stated plainly: **this correction changes the verdict from
NO_INSTRUMENT to a substantive one, which is the self-serving direction, and a reader is entitled to
weigh it accordingly.** What limits the concern is that the error is in the stated sign of a
criterion rather than in the design, that the code change is `point >= floor` becoming
`abs(point) >= floor` with a required sign matching the injected pole, and that both the raw and
orthogonalized numbers are printed at every layer so the corrected verdict can be checked by hand
from the artifact. No other criterion, floor, gate, or control was altered.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
