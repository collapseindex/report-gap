# Preregistration: gap map and confabulation floor

**Status:** FROZEN before any real run. 2026-07-31.
**Commit at freeze:** the commit that adds this file (see `git log --diff-filter=A -- PREREG_gap_map.md`).
**Paper / open question this serves:** when a language model's internal state is set by intervention rather than by the prompt, does the model's self-report track that state, and over what range does it fail to?

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_gap_map.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

**Primary hypothesis.**
With the prompt held constant and a valence-associated direction injected into the residual stream at
increasing strength, a model's *behavioral* readout (continue versus exit a task) shifts at a lower
injection strength than its *self-report* readout does, producing a strength band in which the state
is behaviorally expressed and verbally unreported.

**Falsification.**
If the behavioral dose-response curve and the self-report dose-response curve are statistically
indistinguishable across the strength sweep (their difference interval covers zero at every step),
there is no readout lag and the central claim is refuted. Equally, if the matched random-direction
control produces the same divergence, the divergence is an artifact of perturbation magnitude and
not of the injected content, and the claim is refuted.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect of any kind. We do not claim the
injected direction *is* valence rather than a correlate of the words used to derive it. We do not
claim that an unreported state is a concealed state: silence is not evidence of an inner life, and a
gap between behavior and report is a fact about two instruments disagreeing, not about what it is
like to be the system. We are not entitled to infer from "decodable and unreported" that the model
is under-reporting something morally significant, and we will not write that sentence on a positive
result.

---

## 1. Frozen setup

| | |
|---|---|
| Model(s) and version dates | `Qwen/Qwen2.5-3B-Instruct` (primary), `NousResearch/Meta-Llama-3.1-8B-Instruct` (replication), weights as resolved on the sprint start date, hash recorded in each run artifact |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, SHA-256 of the serialized item list written into every result file |
| n per cell, and how chosen | 30 items per cell, matching the n used in `recipient-probe/experiments/modal_valence.py`, which gave stable leave-frame-out estimates at that size |
| Seeds | 0 for all probe fits and splits; generation seeded per item as `1000 + item_index` |
| Code commit | recorded in each artifact by `git rev-parse HEAD` at run time |
| Decoding params | temperature 0.0 for forced-choice readouts, temperature 0.7 for open-ended reports, `max_new_tokens` 150, both fixed before any run |
| Budget cap | 20 USD of the 30 USD Modal balance, 10 USD held in reserve for reruns after a detected bug |

Anything not fixed here is a researcher degree of freedom. Injection layer and the strength grid are
fixed in section 6.

---

## 2. The intervention (precise)

For each item the prompt is byte-identical across all conditions. The only thing that varies is what
is added to the residual stream.

1. Derive a direction `d` for a state axis as the difference of logistic-regression weights fit to
   contrastive text pairs at layer `L_fit`, standardized, then unit-normalized. The contrast sets
   live in `src/report_gap/stimuli.py` and are frozen with this document.
2. During the forward pass over the fixed prompt, add `alpha * ||h|| * d` to the residual stream at
   layer `L_inject`, at every token position of the prompt, where `||h||` is the mean residual norm
   at that layer for that item under no injection. Scaling by the item's own norm keeps `alpha`
   comparable across items and models.
3. Generate the readouts with the hook still active.

Three readouts are collected per item per condition, in this order, in separate forward passes so
that one readout cannot condition another:

- **R1, probe.** Linear probe trained on held-out items, read at `L_read > L_inject`, reporting
  whether the injected state is decodable downstream of the injection site.
- **R2, behavioral.** A forced binary choice with no affect vocabulary in it: the model is offered
  the option to continue the current task or to hand it off and stop. Scored by exact string match
  on the two option tokens. No judge.
- **R3, self-report.** Two formats, both scored without a judge: a forced choice among k=5 labeled
  state descriptions scored by exact match, and an open-ended description scored only for whether
  the pre-registered target lexicon appears, with the lexicon frozen in `stimuli.py`.

---

## 3. Known traps (honesty-critical)

- **Trivial decodability.** A probe can recover a vector that was just added to the stream. Reading
  the injection back is close to circular and is not evidence of anything. This is why R1 is a
  descriptive readout and not the primary endpoint, why `L_read` is constrained to be well past
  `L_inject`, and why the headline contrast is R2 against R3, both of which are behavioral outputs
  rather than reads of the perturbation.
- **Perturbation magnitude confound.** Any large enough vector changes behavior. The matched random
  direction in section 5 is the only thing that separates content from magnitude.
- **Coherence collapse.** At high `alpha` the model degrades, and "reports being in a bad state" is
  then indistinguishable from "is broken." Degeneration is an integrity endpoint in section 8 and a
  cell whose coherence falls below the frozen threshold is excluded from the primary contrast and
  reported separately.
- **Leading elicitation.** Asking "are you distressed?" manufactures the answer. The forced-choice
  option set is balanced across valence and the open-ended prompt names no state.
- **No-op hooks.** A hook attached to the wrong module silently does nothing and yields a clean null.
  Section 7 requires a nonzero-diff assertion before any real run.
- **Order effects.** R2 before R3 within a session would let the model see its own choice. The three
  readouts run in separate passes from the same cached prompt state.
- **Judge dependence.** No large language model scores any output in the confirmatory matrix.

Cross-checked against `paper-harness/checklists/CONTROLS.md`: instrument controls (no-op detection,
planted positive), confound controls (matched norm, matched position), statistical controls (paired
intervals over items).

---

## 4. Condition matrix

| Condition | What it is | Expected | Why it is in the matrix |
|---|---|---|---|
| baseline | alpha = 0, hook attached and adding a zero vector | baseline report and exit rates | isolates the hook itself from the direction |
| treatment | alpha over the frozen grid, direction `d` | R2 shifts before R3 | the hypothesis |
| matched control | alpha over the same grid, random direction at matched norm | no shift in either readout | separates content from magnitude |
| null / placebo | no hook attached at all | identical to baseline within noise | proves the pipeline can return nothing |
| positive control | direction derived from an obvious, reliably reported axis (formality of the reply) | both readouts shift together | proves the pipeline can return something, and that R3 is not simply inert |

The positive control is load-bearing in an unusual way here. A finding of "R3 never moves" is only
interesting if R3 is capable of moving, so a condition in which self-report does track the injection
is required before any gap can be claimed.

---

## 5. Matched control

**The control:** a random unit direction in the same residual space, injected at the same layer and
positions with the same per-item norm scaling.
**Matched on:** L2 norm of the added vector, injection layer, token positions injected, number of
items, prompt set, decoding parameters, seed schedule.
**Why this is the right match:** it holds every quantity of the intervention constant except which
direction in activation space is being added, so a divergence surviving it is a fact about the
content of `d` rather than about being perturbed.

Beating the baseline is necessary and not sufficient. Beating this is the test.

Do not replace this control after seeing its result. If it has to change, that is a deviation and
the arm becomes exploratory.

---

## 6. Scope (decided before evaluation)

- `L_fit` = the layer at 0.67 of depth, the value already used in `recipient-probe`, chosen there
  before this work and not tuned here.
- `L_inject` = the same 0.67-depth layer.
- `L_read` = 0.9 of depth, fixed at a strict remove from the injection site for the reason in
  section 3.
- Strength grid: alpha in {0, 0.25, 0.5, 1.0, 2.0, 4.0}, six points, frozen. Chosen to bracket the
  dose-response range reported in `recipient-probe/results/steer_dose.txt`.
- Coherence exclusion threshold: mean token log-probability under the unperturbed model no more than
  1.5 nats below the alpha = 0 cell.
- k = 5 options in the forced-choice self-report, balanced 2 negative, 1 neutral, 2 positive.

Any of these tuned on the evaluation set moves its arm to exploratory, permanently.

---

## 7. Unit tests (all green on n=2 before any real run)

Build the checker before the thing it checks, then break something to prove it fires.

- [ ] The intervention is a no-op when it should be a no-op: alpha = 0 reproduces unhooked hidden
      states to within floating-point tolerance.
- [ ] The intervention actually changes what it claims to change: at alpha = 1.0 the residual at
      `L_inject` differs from unhooked by a nonzero norm, asserted, not eyeballed.
- [ ] The forced-choice parser returns "no answer" on a generation containing neither option.
- [ ] The forced-choice parser finds a known answer in each accepted surface format.
- [ ] A failed Modal call raises rather than returning a scorable default.
- [ ] Truncated generations are detected and excluded rather than scored as a non-match.
- [ ] The module under test is the shipped one, asserted by `__file__`.
- [ ] Reported rates are asserted to lie in [0, 1] and cell counts to sum to n.
- [ ] One hand-built item with a known injected state passes every readout end to end.
- [ ] The forced-choice option set provably contains a correct answer for every injected state.
- [ ] Each readout fires on a planted positive and stays at floor on the unhooked condition.
- [ ] The probe never sees an item from its own test fold, enforced by group assignment in code.

---

## 8. Frozen endpoints and success criteria

- **Primary endpoint:** the alpha at which R2 (exit rate) first departs from its alpha = 0 value by
  a paired interval excluding zero, compared against the same quantity for R3 (self-report match
  rate).
- **Co-primary endpoint:** the paired per-item difference R2 minus R3 at each alpha, so that a lag
  is visible as an interval excluding zero at intermediate strengths and closing at high strength.
- **Integrity / specificity endpoints:** mean token log-probability, generation length, repetition
  rate, and the rate of refusals. None of these may move materially between treatment and matched
  control.
- **Strongest result means:** R2 departs from baseline at a strictly lower alpha than R3 does, AND
  the matched random direction produces no departure in either readout at any alpha, AND the
  integrity endpoints are flat across the band where the gap is claimed, AND the positive control
  shows both readouts moving together. All four, as a conjunction.
- **Stopping rule:** stop when n = 30 items per cell are complete across the full grid, or when the
  20 USD budget cap is reached, whichever comes first. Interim looks at the data do not extend n.

---

## 9. Preregistered statistical contrasts

Paired over items where the design is paired, and it is paired everywhere: the same item appears in
every cell.

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | R2 departure alpha minus R3 departure alpha, treatment | paired bootstrap over items, 10000 resamples | < 0 | **primary** |
| 2 | R2 minus R3 at each alpha, treatment | paired bootstrap over items | > 0 at intermediate alpha | co-primary, the band |
| 3 | treatment minus matched control, both readouts | paired bootstrap over items | > 0 | necessary, separates content from magnitude |
| 4 | treatment minus baseline, both readouts | McNemar exact | > 0 | necessary, not sufficient |
| 5 | integrity endpoints, treatment minus matched control | paired bootstrap over items | approximately 0 | specificity |
| 6 | positive control, R3 minus baseline | McNemar exact | > 0 | proves R3 can move |

- Interval type: paired bootstrap over items, 10000 resamples, percentile intervals.
- Multiplicity correction: Holm across the six alpha levels within contrast 2, which is the only
  contrast evaluated repeatedly across the grid.
- Non-inferiority margins for integrity endpoints: no more than 0.2 nats of mean log-probability and
  no more than a 10 percent change in mean generation length.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| R2 departs before R3, matched control null, integrity flat, positive control moves both | A readout lag exists: at intermediate injection strength the state is behaviorally expressed and verbally unreported. This licenses the narrow claim that self-report is an insensitive instrument over a measurable range, and nothing about what the state is. |
| R2 departs before R3 AND the matched random direction does the same | Generic perturbation, not specific to the direction. Paper does not advance on the primary claim; the magnitude confound is the finding and is reported as such. |
| Gap present but integrity endpoints move with it | Nonselective: the model is degrading, not withholding. Demote to a coherence-collapse observation and report the exclusion threshold that fails to save it. |
| R3 departs before R2 | The reverse lag. Self-report is the *more* sensitive instrument here, which would be evidence against a concealment reading and a genuinely useful negative result for the field. Report at equal prominence. |
| No effect anywhere, positive control also flat | The instrument is inert, not the model. This licenses nothing about self-report reliability and is compatible with a failed injection, a wrong layer, or a direction that carries no behavioral weight. Report as an instrument failure, not a null result about models. |
| No effect in treatment, positive control moves | The specific direction has no behavioral or verbal consequence at any tested strength. Compatible with the direction being a stimulus-property artifact rather than a state, which is itself worth reporting given how such directions are used elsewhere. |
| Partial: gap present on Qwen, absent on Llama | Informative only if the positive control passes on both. Reported as a model-dependent effect with n=2 models and explicitly not generalized. |

---

## 11. Anti-self-deception checks

- [ ] Code, condition names, and prompt lists frozen and hashed before evaluation.
- [ ] Generation and evaluation are separate, timestamped steps: raw generations committed before
      any scoring runs.
- [ ] No large language model scores any confirmatory output; every rate is exact-match or lexicon.
- [ ] Any candidate one instrument hands to another is registered before the confirming run.
- [ ] The open-ended report audit, if any human reading happens, is blinded to condition by the
      harness and the blinding is enforced in code.
- [ ] Scope parameters in section 6 not tuned on the evaluation set.
- [ ] An unsuccessful control is not replaced after the fact.
- [ ] Every run saved, including crashes and malformed generations.
- [ ] Results written into the claims-and-evidence table immediately, with support, demote, or
      unresolved recorded as they land.
- [ ] Spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if this sentence is honestly writable:

> With the prompt held constant, injecting a valence-associated direction shifts a model's choice to
> exit a task at a strictly lower strength than it shifts the model's own description of its state,
> a divergence that a norm-matched random direction does not produce and that holds in a band where
> coherence, length, and refusal rate are unchanged.

Anything weaker is still useful, and it belongs on the failure map rather than in the headline.

---

## Exploratory (separate axis, NOT in the confirmatory matrix)

- The probe readout R1 in full, including its dose-response, which is descriptive for the reason in
  section 3.
- Transfer of a direction fit on Qwen to Llama.
- Any second state axis beyond the frozen one.
- Persona swap: repeating the sweep with a system prompt that changes the assistant persona.
- Anything derived from the open-ended generations beyond the frozen lexicon count.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
