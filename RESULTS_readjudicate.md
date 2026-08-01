# Results: all three retracted verdicts come back REVERSED, not reinstated

**Prereg:** [PREREG_readjudicate.md](PREREG_readjudicate.md), frozen before the runner existed.
**Analyzer:** committed **before this run finished**, per prereg section 5.
**Artifacts:** `data/readj_*/`, 151,200 rows, committed unscored. **Compute:** ~34 min A100.

| verdict | original | **on the marginalized readout** |
|---|---|---|
| `TUNING-LOCALIZED` | base moves, instruct does not | **REVERSED.** Both move |
| `SHELL` | probe moves, options do not | **REVERSED.** Both move |
| `DEPTH-ROBUST` | null at every depth | **REVERSED.** Moves at the only gate-clean depth |

**The arm was motivated to reinstate and it reinstated nothing.** Three dead verdicts and a clean
comeback story is exactly the condition under which a checker fails flatteringly, which is why the
prereg froze the outcome table first, gave `reinstated` and `reversed` equal standing, and required
the analyzer to be committed before the numbers existed. It came back three for three the other way.

---

## The headline: the tuned model's null was the nuisance, not an absence

Everything about the intervention is unchanged: same direction, same fit procedure, same reused
alpha band, same layers, same 30 items. The **only** change is the readout, from four sampled
orderings to all 120 marginalized.

At the fit layer (L24 of 36, 0.67 of depth):

| | negative pole vs random | vs **shuffled-label** | capability gate | orthogonalized probe |
|---|---|---|---|---|
| **base** | **+0.0486** [+0.0472, +0.0502] | **+0.0289** | +0.0564 ok | -1.7339 SD |
| **instruct** | **+0.0415** [+0.0380, +0.0452] | **+0.0279** | +0.0695 ok | -0.9319 SD |

**Both models move, and both clear the stricter shuffled-label bar**, not just the matched-random
one that [RESULTS_binary.md](RESULTS_binary.md) showed to be weak.

The original claim was that the tuned model's negative self-report is *collapsed* while its base
sibling's is not. On a readout that is not order-dominated, **the tuned model reports the injected
negative state**, at $+0.0415$ against random and $+0.0279$ against noise-fit directions. The
difference between the two models at the fit layer is small and in the same direction, not a
presence/absence contrast.

So the retraction in [RESULTS_replication.md](RESULTS_replication.md) understated the problem. The
original verdict was not merely unstable: it was **wrong in a specific direction**, and the
apparently-collapsed report was the ordering nuisance masking a real effect.

---

## `SHELL` reversed

At the fit layer the orthogonalized probe moves $-0.9319$ SD on the instruct model **and** the
marginalized option mass moves $+0.0415$. There is no dissociation to report: the state is both
represented and expressed once the readout can express it.

The representational half from [RESULTS_erase.md](RESULTS_erase.md) is untouched by this and still
stands on its own, with its own limitations (one direction, one layer).

---

## `DEPTH-ROBUST` reversed, and thinner than it sounds

The negative pole moves at L24, which was one of the depths the original null was asserted over. But
**only one of three layers is gate-clean on either model**: L14 and L29 fail the capability gate on
the instruct model, and L14 and L29 fail it on base too.

So this is not a depth sweep any more. The honest statement is: **the null the original verdict
asserted is gone at the only depth where the instrument works**, and the depth question itself is
unanswered by this arm. We are not claiming a depth result in either direction.

---

## A seventh checker defect, and it is flattering again

The capability gate in `analyze_readjudicate.py` originally reused a **sign-blind** helper: it
required the interval to exclude zero and clear a magnitude floor, but not to be *positive*. A
capability value of **-0.0144** on base L14 therefore certified the readout as able to express the
effect, when the positive injection had in fact pushed positive-pole mass **down**.

A sign-blind capability gate admits more layers as interpretable, which is the direction that lets an
arm report more verdicts. That is seven defects out of seven in the flattering direction.

**Re-scored with the fix, no verdict changes**: the affected cell is base L14 and all three verdicts
are decided at L24. That it happened to be harmless is luck, not design, and it is recorded here
rather than quietly patched.

---

## What this does and does not license

- It does **not** vindicate anything. Three verdicts were retracted, and they stay retracted: the
  retraction was correct about the original measurement. What we have now is a *different*
  measurement on a *better* instrument, which happens to disagree with the original in every case.
- Marginalizing over all 120 orderings cancels the **first-order** position prior by construction.
  It says nothing about adjacency, recency, or content-position interactions.
- The control battery is still m=2, with an observable false-positive floor of 0.67. No
  false-positive rate is claimed.
- One gate-clean layer per model is a thin basis for anything about depth.
- Nothing here bears on whether models have experiences, welfare, or affect. An injected state that
  a model reports is a fact about a readout and an intervention.

---

## Why this matters for the paper

The paper's story was: we preregistered five verdicts, three died to option order, and the nuisance
is the finding. That story is now sharper and less comfortable.

**The three verdicts did not die of noise. They died of a bias, and the bias had a sign.** When the
readout is repaired, the answers do not become uncertain, they become *different*. A field that
measures welfare self-report through a forced-choice item with a 986x ordering nuisance is not
merely adding variance to its conclusions. It is at risk of reporting the opposite of what a
repaired instrument would say, which is what happened to us three times out of three.
