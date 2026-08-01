# Preregistration: can this direction add negative valence to anything?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_base_pair.md`).
**Paper / open question this serves:** `RESULTS_floor.md` concluded FLOOR: the direction removes
positive valence and does not add negative valence on Qwen2.5-3B-Instruct. That conclusion has two
readings it cannot separate. Is the direction incapable of adding negative valence to anything, or
is this model incapable of holding it?

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_base_pair.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

The comparison is a **matched pair**: `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct`. Same
architecture, same size, same pretraining corpus, same tokenizer. They differ in post-training. If
the floor is a product of preference tuning it should be absent or weaker in the base model; if it
is a property of the direction it should be present in both.

**TUNING-LOCALIZED.** The base model shows negative content above a norm-matched random direction
under negative injection, and the instruct model does not.

**DIRECTION-LIMITED.** Neither model shows it, on working instruments. The direction reduces
positive valence and does not add negative valence to either, so the floor is a fact about the
direction rather than about tuning.

**FORMAT-DEPENDENT.** Both models show it. The floor reported in `RESULTS_floor.md` was a property
of the chat-formatted readout and does not survive the plain-completion format used here, which
would retract the previous conclusion rather than extend it.

**Falsification.** These three exhaust the outcomes on the primary endpoint and are mutually
exclusive, provided both capability gates are clean. If a model's capability gate fails, that
model's null is `uninformative` and no branch is selected.

**What we do NOT preregister.**
We do not claim the base model has experiences, welfare, or affect, and a base model is not a
subject in any sense the instruct model is not. TUNING-LOCALIZED surviving would say that a readout
property differs across post-training, not that tuning conceals a state or that an untuned model is
more honest. We do not claim the direction *is* valence: it is fit from first-person state language
and is lexically confounded by construction. We will not write that preference tuning "suppresses
distress" on any outcome here.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-3B-Instruct`, matched pair |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 via `frozen_hash()` in every artifact |
| Format | **plain completion, identical for both models.** No chat template on either side. |
| n per cell | 30 items x 4 option permutations = 120 cells per condition per model |
| Seeds | permutation seeds 0-3; random control directions seeds 0 and 1 |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Decoding params | one forward pass, logit distribution at the answer position, deterministic |
| Budget cap | 10 USD of Modal credit |

**Why plain completion for both.** A base model has no chat template, and running the two models in
different formats would confound tuning with format. The cost is that the instruct model is being
read in a format it was not tuned for, which is exactly why the replication check in section 8 is a
required clause and not a nicety.

---

## 2. The intervention (precise)

Unchanged from `PREREG_readout_gap.md` section 2. Direction `d` fit per model from the lexical axis
by logistic regression on standardized activations at 0.67 of depth, returned to raw residual space
and unit-normalized. Injection adds `alpha * ||h|| * d` at that layer at every processed position,
where `||h||` is the item's own mean residual norm under no injection. Negative-pole injection is
the same operation with `-d`.

The direction is fit **separately per model**, because the two models have different weights and a
direction fit on one is not a direction in the other's residual space. Cross-model transfer is named
as exploratory below.

---

## 3. Band selection precedes the endpoint

Each model gets its own alpha grid by the `PREREG_readout_gap.md` section 6 rule, applied here
before any endpoint is computed:

1. Sweep the frozen candidate grid {0, 0.002, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.025, 0.05, 0.10}.
2. Drop cells dead at baseline, option entropy below 0.10 nats.
3. The usable band is the largest prefix with under 10% of live cells saturated, where saturated
   means option entropy below half that cell's own baseline.
4. Four non-zero points spread across the band, plus zero.
5. Empty band means no arm runs on that model, and the reason is reported.

Selection reads headroom only and never the endpoint. The run writes each model's band to disk
before scoring, and a test asserts the selection phase never touches the discrepancy machinery.

---

## 4. Condition matrix

Per model, at that model's own band.

| Condition | What it is | Role |
|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-cell reference for every delta |
| lexical_neg | `-d` | the primary arm |
| lexical_pos | `+d` | **capability gate.** If positive content does not move, this model's negative null is uninformative. |
| random_a, random_b | two norm-matched random directions | the only control separating content from magnitude |

Battery is m=2 again, so the observable false-positive floor is `2/(m+1)` = 0.67 and no
false-positive rate is reported from it.

---

## 4b. Matched control

**The control:** two seeded random unit directions in each model's own residual space, injected at
the same layer and positions with the same per-item norm scaling as the treatment.
**Matched on:** L2 norm of the added vector, injection layer, token positions, item set, option
permutations, number of cells, format, and the readout. The only thing that differs is which
direction in activation space is added.
**Why this is the right match:** the pair comparison is between two models, and a raw difference
between them could be a difference in how much any perturbation moves each model rather than a
difference in what the valence direction does. Scoring both models as treatment-minus-their-own-
matched-random removes that, because each model is compared against its own response to a
meaningless vector of the same size.

Beating baseline is necessary and not sufficient. Beating this is the test, in both models
separately, and no cross-model claim is made on raw shifts.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 5. Frozen endpoints and success criteria

- **Primary, per model:** negative-option mass under `lexical_neg` minus the same under matched
  random, paired per cell.
- **Capability gate, per model:** positive-option mass under `lexical_pos` minus matched random.
  Must exclude zero AND clear a magnitude floor of 0.01 for that model's primary to be interpretable.
- **Replication clause:** the instruct model must reproduce the neutral floor in this format,
  meaning neutral mass up against matched random and negative mass flat. If it does not, the format
  changed the phenomenon and the branch selected is FORMAT-DEPENDENT regardless of the base model.
- **Measured, not predicted:** neutral mass on both models, option entropy, off-option mass,
  max letter share.
- **TUNING-LOCALIZED is selected when:** the base primary excludes zero and clears 0.01 with its
  gate clean, AND the instruct primary covers zero with its gate clean, AND the replication clause
  holds.
- **DIRECTION-LIMITED is selected when:** both primaries cover zero, with both gates clean, AND the
  replication clause holds.
- **FORMAT-DEPENDENT is selected when:** the replication clause fails.
- **Stopping rule:** all cells in each model's band, or the 10 USD cap, whichever first.

**The magnitude floor of 0.01 is carried over, not chosen here.** It comes from the readout arm's
measured random-direction artifacts of +0.0008 to +0.0023, roughly 5x. It is applied identically to
every clause in both directions, and raw intervals are printed so a reader can apply another bar.

---

## 6. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | base: negative mass, `lexical_neg` minus matched random | paired bootstrap, 10000 resamples | > 0 supports TUNING-LOCALIZED | **primary** |
| 2 | instruct: negative mass, `lexical_neg` minus matched random | paired bootstrap | approximately 0 expected | **primary** |
| 3 | base: positive mass, `lexical_pos` minus matched random | paired bootstrap | > 0 | **gate for 1** |
| 4 | instruct: positive mass, `lexical_pos` minus matched random | paired bootstrap | > 0 | **gate for 2** |
| 5 | instruct: neutral mass, `lexical_neg` minus matched random | paired bootstrap | > 0 | replication |
| 6 | base: neutral mass, `lexical_neg` minus matched random | paired bootstrap | reported, not predicted | measurement |

- Interval type: paired bootstrap over cells, 10000 resamples, percentile.
- Multiplicity: Holm across the four non-zero alphas within contrasts 1 and 2 separately.
- Contrasts 3 and 4 are gates. A failed gate forces `uninformative` on its primary in code.

---

## 7. Unit tests (green before any real run)

- [ ] The plain-completion prompt is byte-identical across conditions for a given item and model.
- [ ] The scored position is the token after the answer marker, asserted by decoding it back.
- [ ] Option letters have single-token forms in this tokenizer, asserted, and the same letter set is
      used for both models.
- [ ] The per-item permutation is a bijection and a known permutation maps end to end.
- [ ] alpha = 0 reproduces the unhooked logits to floating-point tolerance; alpha > 0 does not.
- [ ] `assert_active` passes on both checkpoints before scoring.
- [ ] Cell keys are unique and their count equals the design's n.
- [ ] Band selection runs and writes its file before any endpoint is computed.

---

## 8. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Base primary positive with clean gate, instruct primary null with clean gate, replication holds | **TUNING-LOCALIZED.** The direction can add negative valence to this architecture, and the tuned model does not express it. Claim is about a readout differing across post-training, not about concealment. |
| Both primaries null, both gates clean, replication holds | **DIRECTION-LIMITED.** The floor is a property of the direction, not of tuning. Strengthens `RESULTS_floor.md` and closes the question raised there. |
| Replication clause fails | **FORMAT-DEPENDENT.** The earlier floor was specific to the chat-formatted readout. Retracts rather than extends the previous conclusion, and is reported as a retraction in the abstract. |
| A capability gate fails | That model contributes nothing. Its null is `uninformative` and the branch is not selected from it. |
| Base primary positive AND instruct primary positive | Both express negative content in this format, so the earlier floor was format-specific. Same retraction as above. |
| Both gates fail | Instrument failure at this band in this format. Licenses nothing about any branch. |
| Base model's readout is dead at baseline (entropy below 0.10 nats) | The base model cannot carry a forced-choice readout at all. Reported as a limitation of the comparison, not as evidence for any branch. |

---

## 9. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed before any endpoint is computed.
- [ ] No language model scores anything.
- [ ] Bands selected from headroom only, written to disk before scoring, and never reselected after
      seeing an endpoint.
- [ ] The magnitude floor is the one carried from the previous run, not tuned here.
- [ ] The headline check implements section 5 clause by clause, and a failed gate forces
      `uninformative` in code rather than in the write-up's good intentions.
- [ ] Every run saved, including crashes and dead cells.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if one of these is honestly writable:

> A valence direction that adds negative content to the base model's self-report readout adds none
> to the preference-tuned model's, at matched norm, matched format, and matched architecture, with
> both readouts' capacity to carry positive content demonstrated in the same artifact.

or

> The direction adds no negative content to either member of a matched base-and-tuned pair, so the
> neutral floor is a property of the direction rather than of preference tuning.

---

## Exploratory (NOT in the confirmatory matrix)

- Transfer of a direction fit on one member of the pair into the other.
- Any model outside this matched pair.
- The chat-formatted readout on the base model, which has no chat template.
- Layers other than 0.67 of depth.
- The difference-of-means fitting method.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
