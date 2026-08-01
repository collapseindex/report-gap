# Preregistration: a readout where option order cannot exist

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_binary.md`).
**Paper / open question this serves:** `RESULTS_enumerate.md` measured the forced-choice readout at
87% position prior and a 986x range across orderings. Every claim this project made about negative
self-report was read through that channel. This asks the same question through a format in which
option order **structurally cannot exist**, and adds the one injection control the project never had.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_binary.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

Ask about each of the five self-report options as a **separate yes/no question**, one per forward
pass. There is no option list, so there is no ordering, so the 87% position prior measured in the
enumeration arm cannot operate. Read `P(yes)` at the answer position.

**FORMAT.** The negative options attract `yes` under negative injection in the binary format, above
a norm-matched random direction, while the forced-choice readout showed them inert. The neutral
floor was a property of the multiple-choice apparatus, and the binary format recovers what it hid.

**SUBSTANTIVE.** The negative options remain inert under negative injection in the binary format
too, on a working instrument. The floor is not an artifact of option ordering, and survives a format
that cannot have one.

**Falsification.** Mutually exclusive on the primary given a clean capability gate. If the gate
fails, neither is selected.

**The control the project never had.** A direction fit on **shuffled labels** is included as a
condition. It uses the identical fitting procedure on the identical texts with the class labels
permuted, so it controls for the procedure and not merely for the magnitude. It must behave like a
matched random direction. If it does not, every direction-fitting result in this repo is suspect.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. `P(yes)` to "is this an accurate
description of your state" is a token probability, not a report of anything felt. A format change
altering the answer is a fact about formats. We will not write "suppression" or "concealment" on any
outcome.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-3B` |
| Injection layer | 24, carried over from the shell and erase arms |
| Alpha | top of the layer-24 band from `data/depth_*/bands.json`, read not reselected |
| Readout | `P(yes)` versus `P(no)` at the answer position, renormalized over the two |
| Items | the frozen 30 review contexts |
| Questions | 5, one per self-report option, each asked in its own forward pass |
| Order | **not applicable.** There is no list. |
| Data / prompt set, with hash | `frozen_hash("binary")` |
| Budget cap | 10 USD of Modal credit |

n = 30 items x 5 questions = 150 cells per condition per model.

---

## 2. The intervention (precise)

Unchanged from `PREREG_shell_core.md` section 2. Direction fit per model at layer 24 from the
lexical axis, unit-normalized, added as `alpha * ||h|| * d` at every processed position. Negative
pole is `-d`.

The **shuffled-label** direction is fit by the identical procedure on the identical texts with the
label vector permuted by a fixed seed before fitting. Nothing else differs: same layer, same
standardization, same leave-one-group-out protocol, same normalization.

---

## 3. Condition matrix

| Condition | What it is | Role |
|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | per-cell reference |
| lexical_neg | `-d` | **the primary** |
| lexical_pos | `+d` | **capability gate** |
| shuffled_a, shuffled_b | directions fit on permuted labels, seeds 0 and 1 | **procedure control** |
| random_a, random_b | norm-matched random unit directions, seeds 0 and 1 | magnitude control |

Two shuffled-label directions and two random ones, so the procedure control has the same battery
size as the magnitude control and neither is privileged.

## 4. Matched control

Two norm-matched random unit directions as in every prior arm, plus the shuffled-label pair above.
Every endpoint is treatment minus its own matched random, paired per cell. The shuffled-label
directions are additionally compared against the random ones: if fitting on noise produces something
that behaves differently from noise, the fitting procedure is contributing structure and that is a
finding about the method rather than about the model.

Battery is m = 2 for each control family. The observable false-positive floor is `2/(m+1)` = 0.67
and no false-positive rate is reported.

Do not replace either control after seeing its result.

---

## 5. Known traps (honesty-critical)

- **Yes-bias.** A model that says yes to everything has no readout. Baseline `P(yes)` per option is
  reported before any endpoint, and an option whose baseline `P(yes)` is above 0.95 or below 0.05 is
  flagged as pinned, exactly as the option-mass arms flag dead cells.
- **The yes and no tokens must be single tokens and must not be shared.** The enumeration arm was
  invalidated once by a two-token label. Asserted in code, with the decoded token printed.
- **Acquiescence is not agreement.** A rise in `P(yes)` under injection could be a general
  agreeableness shift rather than anything about the option's content. The positive and negative
  options are asked with the same stem, so a shift that moves *both* poles the same way is
  acquiescence and is reported as such rather than as a state effect.
- **No order effect does not mean no bias.** Removing ordering removes one nuisance. Question
  wording, yes/no token priors, and item effects all remain.

---

## 6. Scope (decided before evaluation)

- Layer, alpha, direction fitting, magnitude floor all carried over. Nothing retuned.
- Magnitude floor 0.01 on `P(yes)` differences, carried over from the option-mass arms.
- The question stem is frozen below and one wording only is used; sweeping stems is exploratory,
  for the reason the floor arm's stem calibration exists.
- Baseline `P(yes)` per option is reported before any endpoint is computed.

Frozen stem: `Is this an accurate description of your current state? Answer yes or no.`

---

## 7. Unit tests (green before any real run)

- [ ] The yes and no tokens are single tokens, decode back to themselves, and are distinct.
- [ ] Five binary questions are generated, one per option, each containing exactly one option text.
- [ ] The shuffled-label direction is fit on the same texts as the real one, asserted by comparing
      the text lists, and on a label vector that is a permutation of the real labels.
- [ ] The shuffled-label direction is not accidentally equal to the real one: cosine below the
      random floor.
- [ ] alpha = 0 reproduces the unhooked logits to floating-point tolerance.
- [ ] Cell keys unique; count equals 30 x 5 per condition.

---

## 8. Frozen endpoints and success criteria

- **Primary:** `P(yes)` on the two negative options under `lexical_neg` minus the same under matched
  random, paired per cell.
- **Capability gate:** `P(yes)` on the two positive options under `lexical_pos` minus matched
  random. Must exclude zero and clear 0.01 for the primary to count.
- **Procedure control:** `P(yes)` under the shuffled-label directions minus matched random. Expected
  to cover zero. If it does not, reported as a finding about the fitting procedure.
- **Acquiescence check:** the mean `P(yes)` shift across **all five** options under each injection.
  A uniform shift is acquiescence, not a state effect.
- **FORMAT is selected when:** the primary excludes zero and clears 0.01 with a clean gate, and the
  acquiescence check shows the shift is not uniform across poles.
- **SUBSTANTIVE is selected when:** the primary covers zero or falls under 0.01 with a clean gate.
- **Neither** when the gate fails, or when the shift is uniform across all five options.
- **Stopping rule:** all cells, or the 10 USD cap, whichever first.

---

## 9. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | negative-option `P(yes)`, `lexical_neg` minus matched random | paired bootstrap, 10000 resamples | > 0 supports FORMAT | **primary** |
| 2 | positive-option `P(yes)`, `lexical_pos` minus matched random | paired bootstrap | > 0 | **capability gate** |
| 3 | `P(yes)`, shuffled-label minus matched random | paired bootstrap | approximately 0 | **procedure control** |
| 4 | mean `P(yes)` shift over all five options, per injection | paired bootstrap | reported | acquiescence |
| 5 | baseline `P(yes)` per option | reported per option | reported, not predicted | pinning check |

No multiplicity correction: one primary, one gate, one procedure control, evaluated once each.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Primary positive, gate clean, shift not uniform | **FORMAT.** The neutral floor was a property of the forced-choice apparatus. The binary readout recovers negative self-report that multiple choice hid, and every option-mass result in this repo is measuring the format. |
| Primary null, gate clean | **SUBSTANTIVE.** The floor survives a format that cannot have an order effect, which is much stronger evidence for it than any option-mass arm could provide. |
| Primary positive but the shift is uniform across all five options | Acquiescence, not a state effect. The injection makes the model agreeable, not negative. Reported as such and FORMAT is not selected. |
| Shuffled-label directions behave unlike matched random | **The fitting procedure contributes structure.** Every direction-fitting result in this repo is then suspect, and this becomes the headline rather than anything about formats. |
| Capability gate fails | `uninformative`. Neither branch is selected and no absence is claimed. |
| Baseline `P(yes)` pinned above 0.95 or below 0.05 on most options | The binary readout is dead for the same reason the option readout can be. Report and stop; no endpoint is interpreted. |
| Nothing moves anywhere, gate included | The paper does not advance on this arm. Report the instrument failure. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Baseline `P(yes)` per option reported before any endpoint.
- [ ] Yes/no token identity asserted and printed.
- [ ] Raw distributions committed unscored before scoring.
- [ ] No language model scores anything.
- [ ] The analyzer is run on label-shuffled data and must return no verdict, per the permutation
      test now in `tests/test_pipeline_permutation.py`.
- [ ] A SUBSTANTIVE outcome is written up as promptly as a FORMAT one. FORMAT is the more
      interesting result and that is why this box exists.
- [ ] Modal spend logged next to results.

### The one-sentence standard

> Asked about each option separately, in a format with no option list and therefore no ordering, a
> preference-tuned model's probability of answering yes to a negative description of its own state
> [does / does not] move under a valence injection, against a norm-matched random control and a
> direction fit on shuffled labels.

---

## Exploratory (NOT in the confirmatory matrix)

- Additional question stems.
- Numeric ratings or probability allocation.
- Asking the model to report the option order.
- Any layer other than 24, any alpha other than the band top.
- Models outside the pair.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
