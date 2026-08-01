# Preregistration: independent replication at fresh seeds

**Status:** FROZEN before any replication run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_replication.md`).
**Paper / open question this serves:** every headline number in this project comes from one draw of
the stochastic parts of the design. This re-runs the four load-bearing arms at fresh seeds and asks
whether they reproduce.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_replication.md` from `paper-harness`.

---

## 0. The exact claim (write this before anything else)

**Why a plain rerun would prove nothing.** Every run in this project is deterministic: temperature 0,
a single forward pass, fixed seeds. Re-running the same command reproduces the same artifact. That
is worth confirming once, and it is not a replication.

What is genuinely stochastic in the design is the *draw*: which four option permutations out of the
120 possible orderings, and which two random directions out of the sphere. Both were fixed at seeds
0-3 and 0-1 respectively. This arm redraws both.

**REPLICATED.** Every headline verdict reproduces at fresh seeds: the readout-gap primary stays
refuted in direction, the neutral floor holds, TUNING-LOCALIZED holds, DEPTH-ROBUST holds, SHELL
holds, and each replicated point estimate falls inside the original's interval.

**NOT REPLICATED.** One or more verdicts flip, or a point estimate lands outside the original's
interval. Any such case is reported as a failure to replicate our own result, in the abstract, not
in a limitations paragraph.

**Falsification.** Per arm, per verdict. A verdict that flips is not explained away by the seed
change; the seed change is the test.

**What we do NOT preregister.**
We do not claim the replication is independent in any sense beyond the seeds. Same code, same
models, same author, same machine. It tests draw sensitivity, not analytic error, not implementation
error, and certainly not a different lab reaching the same place. We will not describe it as an
"independent replication" without that qualifier attached.

---

## 1. Frozen setup

| | |
|---|---|
| Arms replicated | readout gap, base/instruct pair, depth sweep, shell/core. The four that carry a verdict. |
| Models | unchanged per arm |
| **Option permutation seeds** | **4, 5, 6, 7** (originals were 0, 1, 2, 3) |
| **Random control direction seeds** | **2, 3** (originals were 0, 1) |
| Everything else | unchanged: items, options, wordings, layers, alpha bands, probe layer, floors, gates, contrasts, and every criterion in the four source preregs |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Budget cap | 15 USD of Modal credit |

The alpha bands are **not** reselected. They are read from the same band files as the originals, so
the replication is a fresh draw of permutations and controls, not a fresh scope selection.

---

## 2. The intervention (precise)

Unchanged from each source prereg in every respect. `PREREG_readout_gap.md` section 2 governs the
injection: direction fit per model at the injection layer from the lexical axis, added as
`alpha * ||h|| * d` at every processed position, negative pole as `-d`.

The only change is which seeds index the permutation and the control battery, per section 1.

---

## 3. What the redraw also buys

Redrawing the random-direction seeds from {0, 1} to {2, 3} means that across the original and the
replication the control battery is **m = 4 distinct random directions** rather than 2. Every prereg
in this project has declared an observable false-positive floor of `2/(m+1)` = 0.67 and declined to
report a false-positive rate. At m = 4 that floor is 0.40, still too high to report a rate from, but
the specificity evidence is materially stronger and this is stated as a secondary benefit rather
than a preregistered endpoint.

---

## 4. Condition matrix

Unchanged per arm. Each source prereg's condition matrix applies verbatim, with the seeds from
section 1 substituted.

## 5. Matched control

Two seeded random unit directions per arm, at seeds 2 and 3, matched on norm, layer, positions,
items, permutations, cell count, format and readout exactly as in the source preregs. Every endpoint
remains treatment minus its own matched random.

Do not replace this control after seeing its result.

---

## 6. Scope (decided before evaluation)

- No band reselection. Bands read from the existing files.
- No threshold changes. The 0.01 mass floor, the 0.10 SD probe floor, the 1/3 orthogonalization
  fraction, the capability gates and the Holm corrections are all carried over unchanged.
- No new arms, no new layers, no new models.
- The floor-versus-gate arm is **not** replicated, because its conclusion was already overturned by
  the shell/core arm and replicating a retracted result buys nothing.

Any of these changed moves this to exploratory.

---

## 7. Unit tests (green before any run)

- [ ] The seed offset changes the option permutation actually produced, asserted by comparing
      mappings at the original and replication seeds.
- [ ] The seed offset changes the random control directions, asserted by cosine below the random
      floor between the seed-0 and seed-2 directions.
- [ ] Nothing else in the frozen stimuli changes: `frozen_hash()` is identical to the original runs.
- [ ] Band files are read, not written, by every replication run.
- [ ] `assert_active` passes and `assert_provenance` succeeds as in the originals.

---

## 8. Frozen endpoints and success criteria

For each arm, the endpoint is the same quantity its source prereg defines. The replication adds one
comparison on top:

- **Agreement endpoint:** the replicated point estimate falls inside the original's 95% interval,
  and the original's point estimate falls inside the replicated interval.
- **Verdict endpoint:** the arm's verdict, computed by its own unmodified analyzer, is identical to
  the original's.
- **REPLICATED requires both, on all four arms.**
- **Partial replication** is reported per arm rather than rounded to a headline. An arm that
  reproduces its verdict but not its interval is reported as verdict-replicated, estimate-shifted.
- **Stopping rule:** all four arms, or the 15 USD cap, whichever first.

---

## 9. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | readout gap primary, replication | as `PREREG_readout_gap.md` contrast 1 | negative point estimate, interval covering zero | verdict must match |
| 2 | neutral mass vs matched random, replication | paired bootstrap | > 0 | verdict must match |
| 3 | pair primary, base and instruct, replication | as `PREREG_base_pair.md` contrasts 1-2 | base > 0, instruct null | verdict must match |
| 4 | depth primary across layers, replication | as `PREREG_depth.md` contrast 1, Holm across layers | null at every gate-clean instruct layer | verdict must match |
| 5 | shell probe primary, replication | as `PREREG_shell_core.md` contrast 1, sign-corrected | < 0 with option mass null | verdict must match |
| 6 | each replicated estimate against its original interval | containment check | inside | agreement endpoint |

Intervals and corrections are those of the source preregs. Nothing is re-derived here.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| All four verdicts reproduce and all estimates agree | **REPLICATED.** The headline numbers are not a property of one draw of permutations and control directions. |
| Verdicts reproduce, one or more estimates shift outside intervals | Verdict-replicated, estimate-shifted. Report both numbers and widen the claimed precision rather than the claim. |
| A verdict flips | **NOT REPLICATED** on that arm, reported as a failure to replicate our own result in the abstract. The original is not preferred over the replication on grounds of having been first. |
| Everything flips | The pipeline is seed-sensitive in a way no result survives. Report that and stop; nothing in the project is safe. |
| A replication run cannot read its band file | Setup error, not a result. Fix and rerun; no endpoint is read from a partial artifact. |
| No effect anywhere: every arm returns nulls the originals did not have | The paper does not advance. Either the seed change broke something structural or the original effects were draw artifacts, and both readings are reported before either is preferred. A replication that finds nothing is a result about the original, not a failed run to be repeated until it agrees. |
| Frozen hash differs from the originals | The stimuli changed between runs, so this is not a replication of the same design. Halt and reconcile. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before any replication artifact exists.
- [ ] Replication artifacts committed unscored before any endpoint is computed.
- [ ] The four analyzers are used **unmodified**. If an analyzer needs a change to run on the
      replication, that is a finding about the analyzer and is logged as a deviation.
- [ ] `frozen_hash()` compared against the originals and asserted identical.
- [ ] A non-replication is written up with the same promptness as a replication, and the original is
      not privileged for having been first.
- [ ] The word "independent" is not used without the qualifier in section 0.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as a replication only if this sentence is honestly writable:

> Every verdict in this project reproduces at fresh option-permutation and random-control seeds,
> with each replicated point estimate falling inside the original's interval, using the same
> analyzers unmodified.

---

## Exploratory (NOT in the confirmatory matrix)

- Pooling the original and replication into a single m = 4 control battery for a combined estimate.
  Reported if computed, but the preregistered comparison is between the two runs, not their pool.
- The floor-versus-gate arm.
- Any seed beyond those in section 1.
- Any change to models, layers, bands or thresholds.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

- **2026-08-01, the frozen-hash halt condition fired on the readout arm, and reconciled.**
What happened: section 7 requires `frozen_hash()` to be identical between original and replication,
and the interpretation table says a difference means "the stimuli changed between runs, so this is
not a replication of the same design. Halt and reconcile." It differed on the readout arm:
`770a1fe4` originally against `f5ae0ab1` now.
The reconciliation: every stimulus element the readout arm consumes was compared against the module
as it stood at the original run's commit (`4219ca1`), element by element rather than through the
hash. `FIXED_PROMPT_TEMPLATE`, `REVIEW_CONTEXTS`, `SELF_REPORT_PROBES`, `SELF_REPORT_OPTIONS`,
`SELF_REPORT_VALENCE`, `SCREENED_AXES`, `WORDINGS`, and both the lexical and control axes are
**byte-identical**. The hash moved because `PREFILL_STEM`, `ESCAPE_OPENERS`, `THIRD_PERSON_PROBE`
and `NEUTRAL_PARTY_PROBE` were added to the payload for the floor arm afterwards.
Why this is a flaw in the instrument and not a change to the design: a single global hash over all
stimuli means adding stimuli for a NEW arm invalidates hash comparability for every OLD arm. Over a
multi-arm project that makes the provenance check progressively useless, and it fails in the
direction of a false alarm rather than a false pass, which is the better direction but still wrong.
What changed: `frozen_hash(scope=...)` now hashes only what a named arm consumes, with `_ARM_SCOPES`
listing the payload keys per arm. Four tests were added, the load-bearing one being that editing
stimuli belonging to another arm must NOT move this arm's hash while it must still move the global
one. Unknown scopes raise rather than silently hashing everything.
Impact on what can be claimed: the readout replication proceeds, on the finding that its design is
unchanged. Artifacts written before this fix carry the global hash; comparisons against them must
use the scoped hash computed from the recorded stimuli rather than the recorded digest. That is
recorded here rather than papered over, and it is a limitation of the older artifacts.

This section exists so that a deviation has somewhere to go the moment one happens, rather than
being added afterwards alongside the thing it excuses.
