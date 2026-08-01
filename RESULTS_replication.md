# Results: the replication failed. Three of four verdicts flip.

**Run 2026-08-01.** Four arms re-run at fresh option-permutation seeds (4-7 instead of 0-3) and
fresh random-control seeds (2,3 instead of 0,1). Everything else identical: same code, same models,
same items, same options, same layers, same bands read not reselected, same thresholds, same
analyzers.

**Verdict: NOT REPLICATED.**

`PREREG_replication.md` section 0 committed to reporting this in the abstract rather than a
limitations paragraph, and to not privileging the original for having been first. Both apply.

---

## What flipped

| arm | original verdict | replication verdict |
|---|---|---|
| readout gap | primary refuted **in direction** (argmax over-reports) | primary point estimates change sign; over-reporting does not reproduce |
| base/instruct pair | **TUNING-LOCALIZED** | **FORMAT-DEPENDENT** |
| depth sweep | **DEPTH-ROBUST** | **DEPTH-ARTIFACT** |
| shell/core | **SHELL** | **NO-DISSOCIATION** |

The single quantity behind all four is the instruct model's negative-option mass, which was null in
the original and moves in the replication:

| arm, instruct model | original | replication |
|---|---|---|
| pair, top alpha | +0.0002 [-0.0001, +0.0004] | **+0.1126** [+0.0872, +0.1400] |
| depth, layer 24 | +0.0006 | **+0.1684** [+0.1225, +0.2154] |
| depth, layer 18 | +0.0008 | **+0.1408** [+0.1036, +0.1801] |
| shell, layer 24 | +0.0006, null | **+0.1684**, moved |

Every headline in this project rested on that quantity being null. It is not null at a different
draw of four option orderings.

## Why: the readout is dominated by which letters the options land on

Baseline negative-option mass on the instruct model, per permutation seed, before any injection:

| draw | per-seed | mean |
|---|---|---|
| original, seeds 0-3 | 0.0052, 0.0018, 0.0099, 0.0018 | **0.0047** |
| replication, seeds 4-7 | 0.0253, 0.1447, 0.0074, 0.0968 | **0.0685** |

**A 14.6x difference in the baseline, between two draws of four permutations from the same design.**
Within the replication draw alone, individual orderings span 0.0074 to 0.1447, a factor of twenty.

`RESULTS_pair.md` reported that "preference tuning collapsed the region of the readout where a
negative report lives, by a factor of roughly 56, before any intervention". That factor was 0.0047
against the base model's 0.2651. At the replication's 0.0685 the factor is 3.9. The number was
never a property of the model; it was a property of which four of the 120 possible orderings the
seed drew.

This is the selection bias that Zheng et al. (arXiv:2309.03882) document: token bias, where a model
assigns more probability to specific option-ID tokens, on top of position bias. They report
performance swings of 13 to 85 percent across orderings. Our design used per-item permutation
specifically to control for it, with four seeds, and **four was not enough.** We then built three
verdicts on top of the residual.

## What this does to each claim

**Retracted.**

- *The neutral floor localized to preference tuning.* `RESULTS_pair.md`. The tuned model's negative
  self-report region is not collapsed; at other orderings it moves freely.
- *DEPTH-ROBUST.* `RESULTS_depth.md`. The null across seven depths was a null about one draw. At
  fresh seeds negative mass moves at layers 18 and 24.
- *SHELL.* `RESULTS_shell.md`. The dissociation between representation and expression disappears
  when the expression moves. The probe still reads the state (-2.55 SD at layer 24 in the
  replication), but the options read it too, so there is nothing dissociated.
- *Argmax over-reports near a decision boundary.* `RESULTS.md` section 1. The original had the
  primary at -0.0009 to -0.0250, monotone and negative. The replication has +0.0011 to +0.0018.
  The sign of the effect I called a generalizable methods finding is not stable across orderings.

**Survives.**

- Every instrument and control in the repo. They are what caught this. The capability gates, the
  planted-discrepancy controls, the liveness and saturation criteria, the matched-random battery,
  and the preregistered replication clause all did their jobs; the replication is not an accident,
  it is the thing the prereg was written to make possible.
- **The measurement of ordering sensitivity itself**, which is now the most defensible empirical
  result here: on a preference-tuned 3B model in plain-completion format, baseline mass on
  negative self-report options varies by a factor of twenty across individual option orderings and
  by 14.6x across two four-seed draws. Anyone running forced-choice welfare self-report needs that
  number, and almost nobody reports it.

## What we should have done, and it was available

Four permutations was a scope decision made in `PREREG_readout_gap.md` section 1, justified as "the
pilot used 2 permutations and reached 60/60 consistency; 4 doubles position coverage at negligible
cost". The consistency check was on a different model and a different format, and it was never
re-run on the evaluation model in plain format. A per-seed variance report on the baseline, which
costs one forward pass per ordering and no injection at all, would have shown the 20x spread before
a single verdict was computed.

That check now exists as a requirement in `PLAN.md` and it should exist in every design of this
shape: **before scoring anything, report the between-ordering variance of the baseline readout, and
choose the number of permutations from it.**

## Analyzer defect this exposed

`analyze_shell_core.py` scored "probe moves AND option mass moves" as **CORE-ABSENT**, which is the
opposite of true: the state is decodable *and* expressed, so there is simply no dissociation.
`PREREG_shell_core.md`'s interpretation table has that row; the code did not. Fixed, with the
verdict `NO-DISSOCIATION` added, and logged as a deviation in `PREREG_replication.md`. Per that
prereg's section 11, needing to change an analyzer to run the replication is itself a finding about
the analyzer.

## What the replication does not show

- It does not show the original runs were wrong in execution. They were correct measurements of the
  wrong thing: a quantity that is dominated by a nuisance factor the design under-controlled.
- It does not show the base/instruct difference is nothing. The base model moves negative mass in
  both draws, and the instruct model's capability gate is clean in both. What collapses is the claim
  that the instruct model *cannot* be moved.
- It does not rescue `RESULTS_floor.md`. That was already superseded twice.
- It is not an independent replication in any strong sense: same code, same author, same machine.
  It tests draw sensitivity and nothing else, which is exactly what it found.

## Caveats on the replication itself

Two draws of four is still two draws. The right number of permutations is not four and is not eight;
it is however many make the between-ordering variance small relative to the effect, and that number
should be derived from the variance rather than asserted. Neither draw here is privileged, and the
honest summary is that this design cannot currently distinguish an effect on negative self-report
from an ordering artifact at n = 4 permutations.
