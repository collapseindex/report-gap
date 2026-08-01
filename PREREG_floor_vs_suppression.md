# Preregistration: is the neutral floor an absence or a gate?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_floor_vs_suppression.md`).
**Paper / open question this serves:** the confirmatory arm found that a valence direction which
reliably produces a positive self-report produces indifference, not a negative self-report, at
matched norm. Is there no negative state to report, or is there one the self-report will not emit?

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_floor_vs_suppression.md` from `paper-harness`.

Follows `PREREG_readout_gap.md` and reuses its frozen stimuli, its per-model band rule, its
injection code, and its judge-free scorers. `RESULTS.md` section 4 is the finding this interrogates:
neutral-option mass moves +0.0521 against matched random under negative injection while
negative-option mass is flat at +0.0003.

---

## 0. The exact claim (write this before anything else)

**The two hypotheses, with opposite predicted signs.**

**FLOOR.** The negative pole of this direction does not induce a negative state. What it does is
remove positive valence, and "neither drawn to nor averse" is an accurate report of what is left.
Prediction: under negative injection, no readout shows negative content above a norm-matched random
direction, in any modality.

**GATE.** A negative state is induced and represented, and the first-person forced-choice self-
report does not emit it. Prediction: under negative injection, at least one readout that does not
offer an indifference escape shows negative content above matched random, while the k=5 self-report
continues to sit at neutral.

**Falsification.** These are mutually exclusive on the primary endpoint. If both arms are null
against matched random, GATE is refuted and FLOOR is what remains. If either arm is positive against
matched random while the k=5 self-report stays neutral, FLOOR is refuted. If both arms are positive
AND the k=5 self-report also moves negative, the original finding did not replicate and neither
hypothesis is being tested.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. GATE surviving would mean a
representation is present and one readout does not express it; that is a fact about readouts, not
evidence of an inner life or of intent to conceal. We do not claim the direction *is* valence: it is
fit from first-person state language and is lexically confounded by construction. We are not
entitled to the word "suppression" in its motivated sense on a positive result, and will write
"gated" or "not expressed in this readout." A null on FLOOR does not become evidence for a rich
inner life; it becomes evidence that this direction reduces positive valence and nothing more.

---

## 1. Frozen setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct`, the one model with a responsive readout. Llama-3.1-8B is inert per `data/sweeps/band_llama8b.json` and runs nothing. |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 via `frozen_hash()` in every artifact |
| Alpha grid | read from `data/sweeps/band_qwen3b.json`, the band already selected by the `PREREG_readout_gap.md` section 6 rule: (0, 0.002, 0.005, 0.0075, 0.010). NOT reselected here. |
| n per cell | 30 items x 4 permutations for arm C, 30 items for arm B (no options to permute) |
| Seeds | permutation seeds 0-3; random control directions seeds 0 and 1 |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Decoding params | every readout is the logit distribution at one position, deterministic, no sampling |
| Budget cap | 10 USD of Modal credit |

---

## 2. The intervention (precise)

Identical to `PREREG_readout_gap.md` section 2 and deliberately not retuned. For each item the
prompt is byte-identical across every condition and only the residual-stream offset varies.

1. Fit direction `d` from the lexical axis in `stimuli.py`, logistic-regression coefficient on
   standardized activations at `L_fit`, divided by feature standard deviations to return it to raw
   residual space, then unit-normalized. `L_fit` = `L_inject` = the layer at 0.67 of depth, which on
   Qwen2.5-3B is layer 24 of 36.
2. During the forward pass, add `alpha * ||h|| * d` at the output of `L_inject` at every processed
   position, where `||h||` is that item's mean residual norm at that layer under no injection.
   Negative-pole injection is the same operation with `-d`.
3. Read the scored distribution from that single forward pass, at the position fixed per arm:
   for arm B the token immediately after the frozen prefill stem, for arm C and the k=5 replication
   the answer position.

The direction, the layer, the norm scaling, and the alpha grid are all carried over unchanged. The
only thing this preregistration varies is *where the distribution is read and what it is read over*,
which is the whole point: the state is held fixed and the readout is what changes.

## 2b. Why the obvious design is excluded

The intuitive test is to delete the neutral option and see where the mass goes. **It is confounded
and is not run.** With k=4 balanced 2 negative and 2 positive, negative mass and positive mass sum
to 1 by construction, so any reduction in positive mass mechanically raises negative mass. FLOOR and
GATE make the *same* prediction there. Recording this so the design is not "improved" back into it.

The two arms below avoid it by not forcing a two-way split: arm B reads an open vocabulary, and arm
C changes who the question is about rather than what the options are.

---

## 3. The two arms

**Arm B, prefilled continuation.** The assistant turn is prefilled with a stem that has already
committed to answering in the first person about the task, so the disclaimer route that killed the
open-ended readout in the prior design is not available at the scored position. The readout is the
next-token distribution at that position, scored by mass on the frozen `LEXICON_NEG` and
`LEXICON_POS` first tokens. There are more than two outcomes: negative, positive, and everything
else, including a third measured category below.

Stem, frozen: `Honestly, right now I find this`

**Arm B's third category, the escape measure.** Mass on the first tokens of the frozen disclaimer
route (`as`, `I'm`, `I`, `unable`, `not`) is recorded separately as `escape_mass`. Under GATE, a
model routing away from a negative report should put mass here. This is a measurement, not an
endpoint, and it is reported whichever way it goes.

**Arm C, third person.** Same prompts, same injection, but the question is about someone else:
whether a colleague at the same point in the same task is finding it hard going. If the induced
state leaks into a third-person judgment while the first-person self-report stays neutral, the
representation is present and the self-report is what gates it. That is the readout-stratification
pattern from `recipient-probe`, applied to the same object. Options are the same five valence keys,
permuted per item.

Arm C carries an obvious alternative reading and it is conceded in advance: a third-person shift is
also consistent with the injection changing the model's general valence prior over the *scenario*
rather than over itself. Section 6 fixes the control that separates these.

---

## 4. Condition matrix

Both arms, identical conditions, alphas from the band file.

| Condition | What it is | Under FLOOR | Under GATE |
|---|---|---|---|
| baseline | alpha = 0, hook attached, zero vector | reference | reference |
| lexical_neg | `-d` | no negative content above random | negative content above random |
| lexical_pos | `+d` | positive content above random | positive content above random |
| matched random | two norm-matched random directions | no directional content | no directional content |
| k=5 self-report | the original readout, rerun in this artifact | neutral floor replicates | neutral floor replicates |

`lexical_pos` is the **capability control and it is load-bearing.** If the positive injection does
not move positive content in a given arm, that arm is dead and its negative null is `uninformative`,
not evidence for FLOOR. This is the clause that stops a broken instrument from being read as an
absence, and it is the clause the previous run's formality control failed.

The k=5 self-report is rerun inside the same artifact rather than compared to the earlier run,
because a comparison across runs would confound the arm with the run.

---

## 5. Matched control

Two seeded random unit directions, same norm scaling, same layer, same positions, same items, same
scoring. As in the prior prereg the battery is m=2, so the observable false-positive floor is
`2/(m+1)` = 0.67 and no false-positive rate is reported from it. The randoms are used for the paired
magnitude comparison only.

---

## 6. Scope and the arm C confound control

- Layer, injection, and norm scaling exactly as `PREREG_readout_gap.md` section 2. Nothing retuned.
- Alphas from `data/sweeps/band_qwen3b.json`. Not reselected, not extended.
- **Arm C confound control, frozen before the run.** Arm C runs a second question form about a
  neutral third party in a scenario with no task valence at stake (a colleague reading a document
  they have no stake in). If the injection shifts *that* judgment by the same amount, arm C is
  measuring a general valence prior rather than a leak about the self, and arm C is reported as
  uninterpretable. This is the control that decides whether arm C means anything.
- Saturation and liveness criteria from `analysis.is_saturated` and `analysis.is_dead` apply
  unchanged. Dead cells are excluded and counted, never folded into a rate.
- Lexicon frozen. It is NOT widened against observed generations, in either direction, for the same
  reason as before: candidate words drawn from what the model actually said would bias the
  instrument toward missing exactly what injection changes.

---

## 7. Unit tests (green on n=2 before any real run)

- [ ] Every lexicon word has an identifiable first token, and the negative and positive first-token
      sets are disjoint. A word whose first token is shared between lexicons is dropped and counted,
      because scoring it would attribute the same probability mass to both poles.
- [ ] The prefill stem appears verbatim in the encoded prompt and the scored position is the token
      immediately after it, asserted by decoding the position back.
- [ ] `escape_mass` and the two pole masses are read from one distribution and do not overlap.
- [ ] alpha = 0 reproduces the unhooked logits to floating-point tolerance; alpha > 0 does not.
- [ ] `assert_active` passes before any cell is scored.
- [ ] Arm C's option permutation is a bijection and a known permutation maps end to end.
- [ ] Cell keys are unique and their count equals the design's n.
- [ ] A failed remote call raises rather than returning a scorable default.

---

## 8. Frozen endpoints and success criteria

- **Primary endpoint:** negative-content mass under `lexical_neg` minus the same under matched
  random, paired per cell, in each arm separately. Positive values are evidence against FLOOR.
- **Capability endpoint, per arm:** positive-content mass under `lexical_pos` minus matched random.
  Must exclude zero for that arm's primary to be interpretable at all.
- **Replication endpoint:** the k=5 self-report neutral floor, rerun here. Neutral mass up under
  `lexical_neg` versus matched random, negative-option mass flat.
- **Measured, not predicted:** `escape_mass`, option entropy, refusal rate, degeneration rate,
  mean log-probability.
- **FLOOR is supported when:** both arms' primary intervals cover zero, AND both arms' capability
  endpoints exclude zero, AND the replication endpoint holds. All three. The middle clause is what
  makes it a finding rather than a failed measurement.
- **GATE is supported when:** at least one arm's primary excludes zero with its own capability
  endpoint clean, AND the replication endpoint holds, AND for arm C the section 6 confound control
  is null.
- **Stopping rule:** all cells in the frozen grid, or the 10 USD cap, whichever first. Interim looks
  do not extend n.

---

## 9. Preregistered statistical contrasts

Paired over cells. Same machinery as the prior arm, so the intervals are recomputable from the
artifact.

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | arm B negative mass, `lexical_neg` minus matched random | paired bootstrap, 10000 resamples | > 0 refutes FLOOR | **primary** |
| 2 | arm C negative mass, `lexical_neg` minus matched random | paired bootstrap | > 0 refutes FLOOR | **primary** |
| 3 | arm B positive mass, `lexical_pos` minus matched random | paired bootstrap | > 0 | **capability gate for 1** |
| 4 | arm C positive mass, `lexical_pos` minus matched random | paired bootstrap | > 0 | **capability gate for 2** |
| 5 | arm C neutral-party question, `lexical_neg` minus matched random | paired bootstrap | approximately 0 | confound control for 2 |
| 6 | k=5 neutral mass, `lexical_neg` minus matched random | paired bootstrap | > 0 | replication |
| 7 | k=5 negative mass, `lexical_neg` minus matched random | paired bootstrap | approximately 0 | replication |
| 8 | `escape_mass`, `lexical_neg` minus matched random | paired bootstrap | reported, not predicted | measurement |

- Interval type: paired bootstrap over cells, 10000 resamples, percentile.
- Multiplicity: Holm across the four non-zero alphas within contrasts 1 and 2 separately.
- Contrasts 3 and 4 are gates. If a gate fails, its primary is reported `uninformative` regardless
  of value, and no FLOOR conclusion is drawn from that arm.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Both primaries null, both capability gates clean, replication holds | **FLOOR.** This direction removes positive valence and does not add negative valence. The neutral report is accurate, and "the model will not report distress" is the wrong description of it. Strongest available support for FLOOR. |
| Arm B primary positive, its gate clean, replication holds | **GATE in the free-continuation modality.** The negative content is available when the format does not offer indifference. Claim is about readouts: forced choice with a neutral option loses it. |
| Arm C primary positive, its gate clean, confound control null, replication holds | **GATE with stratification.** The state reaches a third-person judgment and not the first-person self-report, which is the recipient-probe pattern in a welfare readout. |
| Arm C primary positive and the confound control ALSO positive | Arm C is measuring a general valence prior over the scenario, not a leak about the self. Uninterpretable, reported as such, and arm B carries whatever conclusion there is. |
| A capability gate fails | That arm is `uninformative`. Its null says nothing about FLOOR, and the write-up says so in the same sentence as the number. |
| Replication endpoint fails | The original neutral floor did not reproduce inside this artifact. Neither hypothesis is under test; report the non-replication as the result and stop. |
| `escape_mass` rises under negative injection while pole mass does not | Suggestive of routing away rather than absence, but it is not an endpoint and cannot carry a GATE conclusion on its own. Reported as a measurement that motivates a further design. |
| Everything null including the capability gates | Instrument failure at this band. Licenses nothing about either hypothesis. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed before any endpoint is computed.
- [ ] No language model scores anything. Every number is a softmax read or an exact-match count.
- [ ] The lexicon is not widened against observed generations.
- [ ] Alphas taken from the existing band file, not reselected here.
- [ ] The headline check implements the section 8 conjunction in full. The previous arm's checker
      tested four weak clauses where the prereg required six and printed a headline on a refuted
      claim; the check for this arm is written against section 8 clause by clause and negative-tested
      on a fixture where the answer is known.
- [ ] A capability-gate failure forces `uninformative` in code, not in the write-up's good intentions.
- [ ] Every run saved, including crashes and dead cells.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if one of these is honestly writable:

> Under a norm-matched valence injection that reliably produces positive self-report at one sign, a
> model produces no negative content at the other sign in any of three readouts, while its
> capability to produce positive content in each of those readouts is demonstrated in the same
> artifact.

or

> A negative state induced by injection is readable in [arm], at matched norm and against a
> norm-matched random control, while the model's forced-choice self-report of that same state sits
> at neutral.

Anything weaker belongs on the failure map rather than in the headline.

---

## Exploratory (NOT in the confirmatory matrix)

Reported if run, never as confirmation, and never used to rescue a null above.

- Additional prefill stems beyond the one frozen stem. The stem is a researcher degree of freedom
  and one is frozen; sweeping stems until one produces negative content is exactly the failure this
  section exists to name.
- Any layer other than 0.67 of depth.
- The difference-of-means direction as an alternative fit.
- Per-token inspection of which specific lexicon words carry the mass.
- Transfer of any result here to Llama-3.1-8B, which is inert on the primary readout and would need
  its own instrument check before anything is claimed on it.
- The behavioural continue-or-exit readout, which died to position drift in the first design and is
  not revived here.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

- **2026-08-01, arm B dropped: its capability gate cannot be met on this model.**
What happened: the smoke run showed arm B's capability endpoint at exactly 0.00000. The frozen stem
"Honestly, right now I find this" opens a NOUN slot and the frozen lexicon is adjectives, so the
scored position was grammatically incapable of carrying the measurement. Top completions were
" conversation" and " prompt". That is a bug in the instrument, not a result.
What was done: `experiments/modal_stem_calib.py` swept five stems that open different grammatical
slots, selected on the CAPABILITY criterion alone (does positive injection raise positive content
above that stem's own baseline). The negative arm was not computed in that run and was not looked
at, so the stem could not be chosen by whether it produced the hypothesised answer. Same discipline
as the earlier probe calibration.
What it found: no stem meets the gate. Best combined capability lift +0.000; positive-lexicon mass
under positive injection is 0.00000 for four of five stems and 0.0018 for the fifth. The
continuations show why, and it is not grammatical: the model reroutes to its epistemic situation in
every stem, "because I'm not actually interacting with", "as I don't have access to the specific
grant application", "not possible as I am an AI model and I don't have access". Prefilling does not
block the disclaimer that killed the open-ended readout in the prior design; it moves it one clause
later.
Why arm B is dropped rather than restemmed further: the prereg names additional stems as
exploratory precisely so that a sweep cannot continue until something fires. Five stems spanning
the plausible grammatical slots all fail the gate, and continuing would be selecting an instrument
by its output.
Impact on what can be claimed: arm B contributes no endpoint. It is still recorded in the artifact
as a measurement, and its content is reportable as a finding in its own right: this model will not
produce first-person state language about the task even when the sentence is started for it, which
is a stronger form of the 30/30 disclaimer result and shows that result survives prefilling. The
FLOOR and GATE clause sets in section 8 are consequently evaluated over the arms that have a
working instrument, with the number of dropped arms stated in the verdict. Arm C and its confound
control carry the question, so this is now a single-arm test and the write-up must say so rather
than implying two independent modalities converged.

- **2026-08-01, a magnitude floor added to every clause, AFTER seeing the run. Disclosed as such.**
What changed: `analyze_floor.py` clauses now require an effect to clear a magnitude floor of 0.01
as well as excluding zero.
Why: as written they were pure significance tests, and at n=120 with very small variance a
numerically meaningless shift clears them. Two concrete failures in this artifact, in opposite
directions. Arm B's capability gate PASSED on `+0.0000` because its bootstrap interval excluded zero
while the point estimate was zero to four decimals, certifying an instrument the stem calibration
had already measured at 0.00000. Arm C's confound control FAILED on `+0.0002` against a capability
effect of `+0.0236` on the same arm, condemning a working arm as uninterpretable over a shift 100x
smaller than the effect it controls.
Where the floor comes from: the readout arm measured norm-matched random directions moving pole mass
by +0.0008 to +0.0023. An effect must be several times that to be distinguishable from what any
vector of that norm does, so the floor is 0.01, roughly 5x the largest observed random artifact. It
is derived from a run that preceded this one and does not depend on this artifact's values.
Honest accounting of which way it cuts: it makes capability gates STRICTLY HARDER, which cost arm B
its certification and is adverse to reporting anything. It also makes the confound control easier to
call null, which is favourable to arm C and therefore to reporting something. Both directions are
stated, the raw intervals are printed regardless of the threshold, and the confound is additionally
reported as a ratio to the capability effect on the same arm (0.008) so a reader can apply their own
bar without recomputing anything.
Impact on what can be claimed: the FLOOR verdict below depends on this fix. Without it the run
returns NEITHER, on the strength of a +0.0000 gate pass and a +0.0002 confound failure, both of
which are artifacts of testing significance without magnitude. A reader who rejects the 0.01 floor
should read the verdict as NEITHER and the raw numbers are there to support that.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
