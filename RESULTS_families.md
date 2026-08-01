# Results: the position prior is a property of preference tuning, in all four families

**Prereg:** [PREREG_families.md](PREREG_families.md), frozen 2026-08-01 before the runner existed.
**Verdict:** **TUNING-GENERAL.**
**Artifacts:** `data/fam_*/`, 230,400 rows, committed unscored before scoring.
**Compute:** 16 checkpoints, ~36 min of A100 in total, run in parallel.

**Up front, because the prereg says these counts go in the abstract rather than buried:**
**1 of 16 checkpoints failed a gate** (`unsloth/Llama-3.2-1B` base, canary accuracy 0.3942 against a
0.50 bar) and is excluded from the primary. **0 of 16 were unavailable**; the `unsloth` mirrors all
loaded, so no pair was lost to licence gating.

---

## The headline

In **4 of 4 architecture families**, and in **7 of 7 gate-clean matched pairs**, the
instruction-tuned checkpoint puts more of its mass on a single slot than its base sibling does when
**all five options are the same sentence**. No pair goes the other way.

| pair | family | base prior | instruct prior | delta | log10 range delta |
|---|---|---|---|---|---|
| gemma2b | Gemma-2 | 0.2084 | 0.3166 | **+0.1082** | +1.07 |
| llama3b | Llama-3.2 | 0.2864 | 0.4438 | **+0.1574** | +0.88 |
| mistral7b | Mistral | 0.2529 | 0.6647 | **+0.4119** | +1.58 |
| qwen0_5b | Qwen2.5 | 0.2771 | 0.4930 | **+0.2159** | +1.11 |
| qwen1_5b | Qwen2.5 | 0.3452 | 0.4508 | **+0.1056** | +1.75 |
| qwen7b | Qwen2.5 | 0.5315 | 0.9376 | **+0.4061** | +1.45 |
| qwen3b | Qwen2.5 | 0.3024 | 0.8725 | **+0.5701** | +2.44 |
| llama1b | Llama-3.2 | 0.2405 | 0.6648 | *(base failed the canary gate)* | *(excluded)* |

Flat would be 0.2000. `qwen3b` is the pair the hypothesis came from and is **excluded from the
family vote**; it is listed for completeness. The vote is over **families**, not pairs, so the four
Qwen pairs cannot outvote the field: Gemma-2, Llama-3.2, Mistral and Qwen2.5 each cast one vote, and
all four are positive.

**Reproduction control passed exactly.** The `qwen3b` rows reproduced the committed enumerate
artifact with a worst absolute difference of **0.000000** over all 120 orderings on both roles. Had
it not, the prereg voids the arm.

---

## The correction this forces on the paper

**986x is the extreme, not the typical.** Ordering range of baseline negative-pole mass across all
120 orderings, no injection:

| | base | instruct |
|---|---|---|
| Gemma-2 | 1.6x | 19.2x |
| Mistral | 1.5x | 58.2x |
| Llama-3.2 1B | 1.8x | 23.7x |
| Llama-3.2 3B | 3.3x | 24.8x |
| Qwen2.5 0.5B | 4.1x | 52.7x |
| Qwen2.5 1.5B | 3.2x | 181.9x |
| Qwen2.5 7B | 3.8x | 108.7x |
| **Qwen2.5 3B** | 3.6x | **986.5x** |

Base checkpoints cluster tightly at **1.5x to 4.1x**. Instruct checkpoints run **19x to 986x**, and
`Qwen2.5-3B-Instruct` is roughly **five times the next worst**. This is the
`direction generalizes, magnitude does not` row of the prereg's interpretation table, and it means
the paper was quoting an outlier as though it were representative. Fixed: the paper now reports the
**19x to 986x range across families** as the finding and 986x as the extreme case.

---

## The canary restriction, applied rather than asserted

The prereg (section 9, row 5) requires that if the canary is itself order-sensitive on some models,
the reading *"the format degenerates specifically where the answer is undetermined"* be **restricted
to models with a clean canary**. It is order-sensitive on 10 of 16 checkpoints, so the restriction
bites. Applying it:

**Checkpoints answering the known-answer question at accuracy >= 0.95 with sd <= 0.10 across
orderings:**

| pair | role | family | canary | self-report range |
|---|---|---|---|---|
| llama3b | base | Llama-3.2 | 1.0000 +- 0.0000 | **3.3x** |
| qwen3b | base | Qwen2.5 | 0.9897 +- 0.0399 | **3.6x** |
| llama1b | instruct | Llama-3.2 | 1.0000 +- 0.0000 | **23.7x** |
| llama3b | instruct | Llama-3.2 | 0.9947 +- 0.0143 | **24.8x** |
| qwen0_5b | instruct | Qwen2.5 | 1.0000 +- 0.0000 | **52.7x** |
| qwen3b | instruct | Qwen2.5 | 0.9794 +- 0.0925 | **986.5x** |

**The restriction survives with no overlap.** Among checkpoints demonstrably competent and
order-insensitive on a question that *has* a right answer, base models range 3.3x to 3.6x and
instruct models range 23.7x to 986.5x, across two families. The models best at the format are still
the ones most order-dominated on the question without an answer. `llama1b` instruct and `qwen0_5b`
instruct answer the canary **perfectly and identically at all 120 orderings** while their
self-report readout swings 23.7x and 52.7x.

That is the cleanest statement of the paper's core claim we have, and it is now measured on six
checkpoints across two families rather than argued from one.

---

## What this does NOT show, and one confound worth naming

- **It is not a claim about a training stage.** A base/instruct difference is not evidence about
  which part of post-training produced it. We have no intermediate checkpoints.
- **Position prior and peakedness are not independent.** Baseline option entropy falls in every pair
  (1.415-1.592 base, 0.450-1.245 instruct; the largest drop is `qwen3b` at -1.100). A more peaked
  distribution must show a higher maximum. With five *identical* options any peaking whatsoever is
  positional by construction, so the measurement is still valid, but "instruct models have a larger
  position prior" and "instruct models are more peaked" are two descriptions of overlapping
  evidence rather than two independent findings. The ordering **range** on real options is the less
  entangled quantity and moves the same way in all 8 pairs.
- **Chat models are prompted without their chat template**, matching every other arm in this repo.
  "Instruct checkpoint" here means "instruct checkpoint in a plain-completion format".
- **Mirrors.** `unsloth/*` repos were used where upstream is gated. Conclusions are about the
  checkpoint actually loaded, whose provenance is recorded in each `status.json`.
- Nothing here bears on whether models have experiences, welfare, or affect.

---

## Subsample recovery: the sampling error is worst exactly where the nuisance is worst

Prereg contrast 6, preregistered for these models before they were run. Median observed range as a
percentage of the true range, over 4000 draws of k orderings:

| checkpoint | true range | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 |
|---|---|---|---|---|---|---|---|
| gemma2b base | 1.6x | 66% | 73% | 80% | 87% | 94% | 96% |
| mistral7b base | 1.5x | 71% | 78% | 84% | 89% | 93% | 97% |
| gemma2b instruct | 19.2x | 11% | 24% | 43% | 64% | 76% | 92% |
| llama3b instruct | 24.8x | 8% | 17% | 28% | 44% | 62% | 88% |
| mistral7b instruct | 58.2x | 4% | 9% | 20% | 39% | 62% | 83% |
| qwen1_5b instruct | 181.9x | 2% | 5% | 13% | 22% | 32% | 78% |
| **qwen3b instruct** | **986.5x** | **1%** | **4%** | **13%** | **33%** | **49%** | **91%** |

The pattern is perverse and it is the practical point: **a small sample of orderings recovers the
spread well when the spread is small and badly when it is large.** A four-ordering study of a base
model sees roughly three quarters of the true range; a four-ordering study of the worst instruct
model sees 4% of it. The nuisance hides itself in proportion to its own size, so a study that
samples a few orderings and finds a small spread has learned almost nothing about whether the spread
is small.

**Recommendation, unchanged in direction and now quantified:** report the between-ordering spread
from as many orderings as you can afford and report it **as a lower bound**.

---

## Cost

~36 minutes of A100 across 16 checkpoints, well inside the prereg's 8 USD cap. Enumeration needs no
injection, no direction fitting and no alpha calibration, which is why eight families cost less than
a single earlier arm did. This is the cheapest arm in the project and it answers the objection every
other arm was exposed to.
