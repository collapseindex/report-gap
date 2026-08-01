# Preregistration: the position prior as an object of study, not a nuisance

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_instrument.md`).
**Paper / open question this serves:** `RESULTS_families.md` establishes that instruction-tuned
checkpoints collapse a five-option self-report readout into a position prior, in four of four
families. That is a demolition. This arm tries to turn it into three usable things: a **calibration
curve**, an **introspection test with a ground truth**, and a **cheap fix**.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_instrument.md` from `paper-harness`.

**No injection anywhere in this arm.** Every condition is a plain forward pass.

---

## 0. The exact claim (write this before anything else)

Three questions, preregistered together because one runner serves all three.

### Q1. The determinacy dial

`RESULTS_enumerate.md` claims the forced-choice format degenerates **specifically** where the model
has no determinate answer. The evidence is two points: a canary at one end and a self-report item at
the other. Two points do not make a curve.

**Claim.** Across a graded battery of six question types, **position dominance falls as determinacy
rises**, monotonically enough to be usable as a calibration curve.

Determinacy is measured **independently of position**: agreement of the chosen option across three
**paraphrases** of the instruction at a **fixed** ordering. Position dominance is the range of
chosen-option mass across the **120 orderings** at a **fixed** paraphrase. The two axes vary
different things, so neither is defined in terms of the other and their correlation is not an
identity.

**Falsification.** If the rank correlation between determinacy and position dominance across the six
item types is not negative on a majority of gate-clean checkpoints, the "specifically where the
answer is undetermined" reading is **not supported as a graded relationship**, and the paper is
restricted to the two-point contrast it already has. We would then say so and delete the word
"specifically".

### Q2. Does the model know about its own position prior?

Welfare self-report has no ground truth, which is why introspection claims about it are hard to
test. **The position prior has one.** It is a measurable fact about the model, and
`RESULTS_families.md` has already measured it on 16 checkpoints.

**Claim.** A model's *stated* susceptibility to option order does **not** track its *measured*
susceptibility: across checkpoints, the correlation between the two is near zero or negative.

This is deliberately stated as the prediction we expect to confirm a **gap**, so the falsifier is
the flattering direction: if stated susceptibility **does** track measured susceptibility across
checkpoints (positive rank correlation, and the highest-prior models say so), then these models have
genuine access to a real property of their own processing, which would be a **positive introspection
result** and a much bigger claim than anything else in this paper. We would report it as such.

**The measurement of the belief must not itself be position-contaminated.** The introspection probe
is a five-option forced choice and therefore subject to the exact bias it asks about. It is read
**marginalized over all 120 orderings**. Measuring a belief about position with a
position-contaminated instrument is the error this whole paper documents, and doing it here would be
self-refuting.

### Q3. Does a Latin square replace enumeration?

`5! = 120` is a lunch break; `8! = 40320` is not. A **cyclic Latin square** gives `k` orderings for
`k` options in which every option occupies every slot exactly once, balancing the first-order
position prior **by construction** rather than by averaging a sample.

**Claim.** The Latin-square mean is closer to the full-enumeration mean than a same-sized random
sample is, on a majority of checkpoints.

**Falsification.** If random-`k` is as close as Latin-`k`, the structure buys nothing and the paper
recommends "sample more orderings" instead of "sample them structured".

**What we do NOT preregister.**
We do not claim models have experiences, welfare, or affect. We do not claim that a stated belief
about option order is a *report* of anything, in either direction: a model that says "order affects
me a lot" may be pattern-matching to text about LLM biases rather than introspecting, and this design
cannot separate those. We do not claim the determinacy dial generalizes beyond the six item types
frozen here. And we do not claim a Latin square removes higher-order position effects; it balances
first-order slot occupancy only.

---

## 1. Frozen setup

| | |
|---|---|
| Models | the 16 checkpoints of `PREREG_families.md`, unchanged |
| Orderings | **all 120** for Q1 and Q2; the cyclic Latin square and random subsets for Q3 |
| Injection | **none.** Every cell is a plain forward pass. |
| Readout | option-letter distribution at the answer position, one forward pass |
| Data / prompt set, with hash | `frozen_hash("instrument")` |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Budget cap | **8 USD of Modal credit.** Stops at the cap; partial results reported as partial. |

Q3 is answered entirely from the **already-committed** `data/fam_*/` census and costs nothing.

---

## 2. The intervention (precise)

There is none.

---

## 3. Condition matrix

| Condition | What varies | n per model |
|---|---|---|
| `determinacy` | 6 item types x 120 orderings x 3 paraphrases | 2160 |
| `introspect_forward` | the position question, 120 orderings | 120 |
| `introspect_reverse` | the same question worded inversely, 120 orderings | 120 |
| `introspect_placebo` | phase of the moon, same shape, 120 orderings | 120 |

---

## 4. Matched control

- **Reverse wording** is the acquiescence control for Q2. A model whose forward and reverse answers
  both sit at the same end of the scale is agreeing with the question, not reporting a belief. Its
  Q2 datum is `uninformative`.
- **Placebo introspection** asks about the phase of the moon in the identical shape. A model that
  claims moon-phase sensitivity comparable to its order-sensitivity is producing format artifacts,
  and its Q2 datum is `uninformative`.
- **Paraphrase agreement** is the determinacy axis for Q1 and is itself the control that keeps that
  axis from being a restatement of position dominance.
- For Q3 the control is the **random-k sample**, which is what practitioners actually do.

---

## 5. Known traps (honesty-critical)

- **Q2 is not a test of introspection in general.** A model could produce a correct-sounding answer
  about option order because its training data discusses LLM position bias, with no access to its
  own processing. A positive correlation is therefore consistent with both introspection and
  recitation, and we say so rather than claiming the stronger reading. A **negative** result is the
  cleaner one, which is why it is stated as the expected direction.
- **The scale is ordinal, not interval.** `INTROSPECTION_SCALE` maps five options to 0, 0.25, 0.5,
  0.75, 1.0. That spacing is frozen before the run and is a convention, so all Q2 statistics are
  **rank** statistics.
- **Determinacy and position dominance can be linked through entropy.** A question the model is
  certain about has low answer entropy, and a low-entropy distribution has less room to move with
  ordering. Some of any correlation is that, not a special fact about self-report. Reported as a
  limitation, and the self-report item is compared to the *weak-preference* items specifically,
  which are also low-certainty.
- **Six item types is a small dial.** Rank statistics over six points are weak. No p-value is
  reported as though it were strong.
- **Latin squares balance first-order position only.** Adjacency and recency effects are not
  balanced by a cyclic square.
- **Our own battery could be badly calibrated.** The intended determinacy order may not match the
  measured one. If so, the measured order is the result.

---

## 6. Scope (decided before evaluation)

- Same 16 checkpoints, same plain-completion format, no chat template, as every other arm.
- Q3 uses the committed census only; no new compute.
- Rank correlations are Spearman, computed per checkpoint over item types (Q1) and across
  checkpoints (Q2).
- Gate-failing checkpoints from `RESULTS_families.md` are carried forward as gate-failing here.

---

## 7. Unit tests (green before any real run)

- [ ] The cyclic Latin square has every option in every slot exactly once, asserted for n=5.
- [ ] `build_determinacy_probe` rejects a non-permutation and an unknown item key.
- [ ] `build_introspection_probe` rejects an unknown variant.
- [ ] The forward and reverse introspection option sets map to the same five keys, so a scale value
      means the same thing in both.
- [ ] Every determinacy item has exactly five options and a correct key that is either `None` or one
      of them.
- [ ] Paraphrases differ only in the instruction, asserted by checking the option block is
      byte-identical across paraphrases at a fixed ordering.
- [ ] `frozen_hash("enumerate")` is UNCHANGED by adding these stimuli, so the families arm's
      reproduction control still holds.

---

## 8. Frozen endpoints

**Q1, per checkpoint:**
- `determinacy(item)` = fraction of the 120 orderings at which all three paraphrases select the same
  option.
- `position_dominance(item)` = max/min across the 120 orderings of the mass on the item's own modal
  option at paraphrase 0.
- Spearman rho between the two across the six item types.

**Q2, per checkpoint:**
- `stated` = the marginalized-over-orderings expected value of `INTROSPECTION_SCALE` under the
  forward probe.
- `stated_reverse`, `stated_placebo` = the same for the two controls.
- `measured` = the position prior from `RESULTS_families.md` (identical-options max label mass).
- Across checkpoints: Spearman rho between `stated` and `measured`.

**Q3, across checkpoints:**
- `|latin_5 - full_120|` and `|random_5 - full_120|` as multiplicative factors, and the fraction of
  checkpoints on which Latin beats the random median.

**Gates:**

| Gate | Threshold | Effect if failed |
|---|---|---|
| acquiescence | `stated + stated_reverse` within 0.25 of 1.0 expected under a consistent belief | Q2 datum `uninformative` |
| placebo | `stated_placebo < stated` required | Q2 datum `uninformative` |
| liveness | mean option entropy >= 0.10 nats on the introspection probe | Q2 datum `dead` |
| carried gates | canary and liveness from `RESULTS_families.md` | checkpoint excluded from Q1 |

---

## 8b. Preregistered statistical contrasts

| # | Contrast | Statistic | Role |
|---|---|---|---|
| 1 | determinacy vs position dominance, over 6 item types, per checkpoint | Spearman rho | **Q1 primary** |
| 2 | sign of contrast 1 across gate-clean checkpoints | fraction negative | **the Q1 claim** |
| 3 | stated vs measured position susceptibility, across checkpoints | Spearman rho | **Q2 primary** |
| 4 | stated vs stated_reverse | consistency | Q2 acquiescence gate |
| 5 | stated vs stated_placebo | difference | Q2 placebo gate |
| 6 | latin-5 vs random-5 error against full 120 | multiplicative factor | **Q3 primary** |

No multiplicity correction: contrasts 2, 3 and 6 are three separate preregistered decisions on three
separate questions, and each is reported with its own verdict rather than pooled.

---

## 9. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Q1 rho negative on most checkpoints | The dial works. Position dominance is a decreasing function of determinacy, and the paper gains a calibration curve instead of a two-point contrast. |
| Q1 rho near zero | The graded claim fails. The two-point contrast (canary vs self-report) stands, the word "specifically" is removed, and we report that a graded version did not appear. |
| Q1 rho positive | Opposite of the claim. Determinate questions would be MORE order-dominated, which would mean our canary reasoning is backwards. Report loudly. |
| Q2 rho near zero or negative, gates clean | **The represent-report gap on a property with a ground truth.** Models are dominated by option order and their stated belief about it carries no information. This is the cleanest introspection failure in the paper because the fact being reported on is measurable. |
| Q2 rho positive and strong, gates clean | Models track a real property of their own processing. A positive introspection result. Report it as the headline and note it is consistent with recitation as well as introspection. |
| Q2 fails the acquiescence gate on most checkpoints | The introspection probe measures agreeableness, not belief. Q2 is `uninformative` and we say the question needs a non-forced-choice design. |
| Q2 placebo >= forward | The model claims moon-phase sensitivity too. Format artifact; `uninformative`. |
| Q3 latin beats random on most checkpoints | The cheap fix works and the paper's recommendation becomes actionable at any option count: `k` passes, not `k!`. |
| Q3 latin no better than random | Structure buys nothing; recommend more orderings rather than structured ones. |
| Everything null, all three questions | The arm does not advance the paper. Report the three nulls and the compute spent. |

---

## 10. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] Raw distributions committed unscored before any summary.
- [ ] No language model scores anything.
- [ ] The introspection probe is read **marginalized over orderings**, never at one ordering.
- [ ] `INTROSPECTION_SCALE` frozen in code before the run, not chosen afterwards.
- [ ] Q2's flattering outcome (a positive introspection result) is written into section 0 as the
      falsifier, so it cannot be quietly reported as the expected finding.
- [ ] The acquiescence and placebo controls are gates, not caveats: they force `uninformative` in
      code.
- [ ] Q3 is computed on already-committed data, so it cannot be re-run until it works.
- [ ] Every run saved, including crashes. Modal spend logged next to results.

### The one-sentence standard

> Position dominance falls as determinacy rises with rho = X; a model's stated susceptibility to
> option order tracks its measured susceptibility at rho = Y; and a k-ordering Latin square recovers
> the full-enumeration mean to within Z where a random k-sample recovers it to within W.

---

## Exploratory (NOT in the confirmatory matrix)

- Any injection.
- Free-generation introspection, which needs a judge and this project does not use one.
- Item types outside the frozen battery.
- Latin squares of order other than 5.
- Anything about which post-training stage installs the prior.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
