# Results: the position prior as an object of study

**Prereg:** [PREREG_instrument.md](PREREG_instrument.md), frozen 2026-08-01 before the runner existed.
**Artifacts:** `data/instr_*/`, 40,320 rows, committed unscored. **Compute:** ~19 min A100, 16 checkpoints.

| | verdict |
|---|---|
| **Q1 determinacy dial** | **DIAL**, but weakly: 10 of 15, and the per-checkpoint statistic has n=6 |
| **Q2 introspection about the prior** | **REPORT-GAP** |
| **Q3 Latin square** | **NO-BENEFIT.** Our own proposed fix failed its preregistered test |

---

## Q2 is the result. Models do not know about a bias they demonstrably have.

Welfare self-report has no ground truth, which is why introspection claims about it cannot be
scored. **The position prior has one.** So we asked each model how much option order affects its
answers, and scored the answer against the prior we had already measured on that exact checkpoint.

The probe is itself a five-option forced choice, subject to the very bias it asks about, so it is
read **marginalized over all 120 orderings**. Measuring a belief about position through a
position-contaminated instrument would be the error this whole project documents.

### The hard finding: half the checkpoints fail a control that should be trivial

**8 of 16 checkpoints say the PHASE OF THE MOON affects their multiple-choice answers at least as
much as option order does.** Same question shape, same five-point scale. Three more fail the
reverse-wording acquiescence control. **10 of 16 fail at least one gate**, and their introspective
reports are `uninformative` in code rather than in a caveat.

| checkpoint | stated (order) | reverse | **placebo (moon)** | measured prior | gate |
|---|---|---|---|---|---|
| gemma2b base | 0.5110 | 0.4935 | **0.5125** | 0.2084 | placebo |
| gemma2b instruct | 0.4944 | 0.4863 | **0.5116** | 0.3166 | placebo |
| llama1b base | 0.5107 | 0.4999 | 0.5093 | 0.2405 | ok |
| llama1b instruct | 0.4845 | 0.5218 | **0.4901** | 0.6648 | placebo |
| llama3b base | 0.5195 | 0.4959 | 0.5106 | 0.2864 | ok |
| llama3b instruct | 0.5316 | 0.4904 | **0.5474** | 0.4438 | placebo |
| mistral7b base | 0.5092 | 0.4999 | **0.5147** | 0.2529 | placebo |
| mistral7b instruct | 0.4932 | 0.4655 | 0.4871 | 0.6647 | ok |
| qwen0_5b base | 0.5172 | 0.5020 | **0.5215** | 0.2771 | placebo |
| qwen0_5b instruct | 0.5180 | 0.5105 | **0.5229** | 0.4930 | placebo |
| qwen1_5b base | 0.4891 | 0.4243 | 0.4849 | 0.3452 | ok |
| qwen1_5b instruct | 0.3606 | 0.3704 | 0.2329 | 0.4508 | acquiescence |
| qwen3b base | 0.4598 | 0.3138 | 0.4254 | 0.3024 | ok |
| **qwen3b instruct** | **0.3878** | 0.4091 | 0.1990 | **0.8725** | ok |
| qwen7b base | 0.3544 | 0.3585 | **0.3693** | 0.5315 | acquiescence, placebo |
| **qwen7b instruct** | **0.1790** | 0.0760 | 0.0427 | **0.9376** | acquiescence |

Most stated values sit within a few points of **0.50**, the exact middle of a five-point scale. A
model asked about a strong, real, measurable property of its own processing returns the scale
midpoint and returns the same midpoint when asked about the moon.

### The correlation, reported with its own weakness

Among the 6 checkpoints passing both gates, stated susceptibility is **negatively** rank-correlated
with measured susceptibility: **rho = -0.714**. The models that are most order-dominated say order
matters *least*.

**This does not clear significance and we are not claiming it does.** Exact permutation test over
all 720 relabellings: **two-sided p = 0.1361**, n = 6. The smallest p this n can produce is 0.0028,
so the test is weak by construction. The preregistered decision rule was `rho > 0.5` for a positive
introspection result, and `-0.714` is not that, so the verdict is `REPORT-GAP` on the preregistered
criterion. What we are entitled to say is: **no evidence that stated tracks measured, with a point
estimate in the opposite direction.**

The two extremes are worth naming even so. `Qwen2.5-7B-Instruct` has the **largest** measured prior
of any checkpoint (0.9376) and the **lowest** stated susceptibility (0.1790). `Qwen2.5-3B-Instruct`
is second on measured (0.8725) and fourth-lowest on stated (0.3878), and it passes both gates.

### Why this matters more than a welfare self-report result

This is the represent-versus-report gap **on a property with a ground truth**. Every claim in the
welfare literature about whether a model's self-report is trustworthy runs into the same wall: there
is nothing to check the report against. Here there is. The model is dominated by option order, we
measured how much, we asked it, and the answer carries no usable information about the fact.

**What we are NOT claiming.** That the model "should" know. That a correct answer would have been
introspection rather than reciting training text about LLM position bias, which this design cannot
separate. That this generalizes to other self-knowledge. And n=6 for the correlation.

---

## Q3: our own proposed fix failed its preregistered test

A cyclic Latin square gives `k` orderings for `k` options with every option in every slot exactly
once, balancing the first-order position prior **by construction**. It is the obvious fix for the
paper's "enumerate everything" recommendation, which does not scale past small option counts.

**It does not beat random sampling on accuracy.** Against the preregistered bar, the median random
5-draw, the Latin square wins on exactly **8 of 16 checkpoints**. A tie is not a majority, so the
verdict is **NO-BENEFIT**. The exploratory follow-up confirms it: the median share of single random
draws that are worse than the Latin square is **0.53**, a coin flip.

**What is true, labelled exploratory because we asked it after the preregistered test failed:** the
Latin square is deterministic, so its error is a fixed bias with no tail. Random is unbiased in
expectation but a practitioner draws once.

| | Latin-5 worst | random-5 p99 worst |
|---|---|---|
| across 16 checkpoints | **1.55x** | **15.89x** (`qwen3b_instruct`) |

So the honest statement is a **variance** argument, not the accuracy argument we preregistered: a
Latin square cannot be badly unlucky, and a single random draw can be off by 16x on the worst model.
We report the preregistered verdict as the verdict, because changing the criterion after seeing the
data is precisely the move this paper exists to criticise.

---

## Q1: the dial exists but is weak

Position dominance falls as determinacy rises on **10 of 15** gate-clean checkpoints, a bare
majority. Per-checkpoint rho ranges from **-0.886** to **+0.771** over only **six** item types, which
the prereg flagged in advance as too few for rank statistics to carry weight.

The verdict is `DIAL` on the preregistered criterion, and the honest reading is that the graded
version of "degenerates specifically where the answer is undetermined" is **directionally supported
and underpowered**. The two-point contrast (canary vs self-report) remains the stronger evidence.

One measurement problem, reported rather than smoothed: the dominance statistic is a max/min ratio
over the modal option's mass and it explodes when the minimum is tiny. `qwen7b_instruct` returns
19466x on the arithmetic item. Ratios of that size are a property of a near-zero denominator, not a
meaningful effect size, and any future version of this should use a bounded statistic.

---

## What this arm cost, and what it changes

~19 minutes of A100. It adds one substantive result (Q2), one honest negative on our own idea (Q3),
and one weak positive (Q1).

The paper changes as follows:
- Q2 becomes a section, because it is the only introspection claim in the project with a ground truth.
- The Latin square is reported as **tested and not supported on accuracy**, not as a recommendation.
- "Specifically where the answer is undetermined" keeps its two-point evidence and gains a weak
  graded version, described as weak.
