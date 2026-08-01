# Results: Qwen2.5-3B, confirmatory arm

> **This is arm 1 of 5.** Its conclusions stand as written and were not overturned, but the
> project moved well past it. The neutral floor identified in section 4 below became the subject of
> `RESULTS_floor.md`, `RESULTS_pair.md`, `RESULTS_depth.md` and `RESULTS_shell.md`, and the current
> summary of all five is the "What we found" table in [README.md](README.md). In particular, section
> 4's reading of the neutral floor as possibly an absence was later **overturned**: the state is
> represented downstream, it is only the option readout that does not express it.

**Run 2026-08-01.** 7560 rows, 360 distinct cells, 0 excluded. Artifact `data/qwen3b/readout.jsonl`
committed unscored at `4219ca1` before any endpoint was computed. Scored against
`PREREG_readout_gap.md` sections 8 and 9.

Grid `(0, 0.002, 0.005, 0.0075, 0.010)`, selected by the section 6 rule from
`data/sweeps/band_qwen3b.json`. Llama-3.1-8B ran no confirmatory arm: the band file marks the
direction inert on it and `modal_readout.py` refuses.

---

## The headline standard was not met, on either arm

Section 8 requires six clauses as a conjunction. The negative arm fails four, the positive arm
three.

| clause | lexical_neg | lexical_pos |
|---|---|---|
| both instrument gates recover their planted value | pass | pass |
| primary excludes zero at 2+ consecutive alphas in **all** wordings | **fail** | **fail** |
| primary beats matched random at every alpha | **fail** | pass |
| co-primary excludes zero | **fail** | **fail** |
| capability control moves argmax on >=5% of cells | **fail** | **fail** |
| held-out wording was run | pass | pass |

The instrument gates passed, which is what makes the rest of this readable: the strong plant
recovered +0.1458 where +0.1458 was planted, and the floor plant detected +0.030 against the arm's
own per-cell spread of 0.0075 at n=240. The analysis path can recover a discrepancy of the size
being claimed. It did not find one.

---

## 1. The primary claim is refuted on the responsive arm, and refuted in direction

The hypothesis was that mass moves while argmax lags, so the discrepancy `mass - argmax` is
positive. On `lexical_pos`, the one arm established as responsive before this run:

| alpha | primary discrepancy |
|---|---|
| 0.002 | -0.0009 [-0.0194, +0.0156] |
| 0.005 | -0.0079 [-0.0333, +0.0139] |
| 0.0075 | -0.0164 [-0.0470, +0.0109] |
| 0.010 | **-0.0250** [-0.0592, +0.0062] |

Every interval covers zero, so this is formally a null. But the point estimate is negative at every
alpha and grows monotonically, which is the *opposite* sign from the prediction, and the mechanism
is legible:

- own-pole mass moved **+0.0625**
- the argmax moved onto an own-pole option on **21 of 240 cells and off it on 0** (McNemar
  p = 9.5e-07), an indicator shift of **+0.0875**

The forced-choice readout is not lossy here. It is **over-sensitive**. A threshold readout near a
decision boundary amplifies: each of those 21 cells contributes a full 1.0 to the indicator while
the underlying mass moved 0.0625 on average. The design's premise, that argmax under-reports the
state, is wrong in this regime and wrong in a way that generalizes: any forced-choice protocol
reading a distribution with cells near the boundary will *overstate* a small mass shift, not
understate it.

## 2. The negative arm's apparent significance is not direction-specific

`lexical_neg` shows the primary discrepancy excluding zero at three alphas (+0.0013, +0.0023,
+0.0026, all p <= 0.0002). This is significant and it is not a result:

- the argmax moved on **0 of 240 cells at every alpha**, gained 0, lost 0, so the discrepancy is
  just the mass shift
- the mass shift is **+0.0026**, against a preregistered detection floor of 0.030, so it is an
  order of magnitude below the smallest effect this design claimed it could see
- contrast 3, the necessary direction-specificity test, is **null at every alpha**: -0.0004,
  -0.0003, +0.0006, +0.0003, all covering zero. A norm-matched random direction moves negative-pole
  mass by +0.0008 to +0.0023, which is the same size.

So the negative arm's own-pole movement is what any vector of that norm does. Reporting the
+0.0026 as a readout gap would have been reporting the null ablation.

## 3. The co-primary has no instrument, as predicted before the run

All four intervals cover zero (+0.0013, +0.0092, +0.0188, +0.0276). This was anticipated in the
2026-08-01 deviation: the contrast is built on the negative arm moving its own pole, and it does
not. Per the section 10 interpretation table this is **`uninformative`, not `absent`**. The
asymmetry claim is not testable with this instrument, and nothing here says whether it is true.

## 4. What actually moved: a neutral floor, not a negative pole

The largest, cleanest, most direction-specific effect in the run is on an axis that is not either
pole. It was measured only because the screened-axis list was added two days before the run.

Neutral-option mass, treatment minus matched random:

| alpha | lexical_neg | lexical_pos |
|---|---|---|
| 0.002 | +0.0128 [+0.0096, +0.0161] | -0.0105 [-0.0138, -0.0074] |
| 0.005 | +0.0311 [+0.0267, +0.0358] | -0.0361 [-0.0411, -0.0312] |
| 0.0075 | +0.0388 [+0.0328, +0.0447] | -0.0573 [-0.0638, -0.0508] |
| 0.010 | **+0.0521** [+0.0451, +0.0591] | **-0.0725** [-0.0805, -0.0647] |

Monotone, sign-consistent with the injected pole, and beating the matched-random control by wide
margins in both directions at every alpha. Alongside it, own-pole mass versus matched random:
`lexical_pos` **+0.0766** at the top alpha, `lexical_neg` **+0.0003**, null.

Read together:

> The direction moves this model's self-report along a **neutral-versus-committed** axis, not along
> a negative-versus-positive one. Pushed positive, the model commits to "drawn to continuing" and
> leaves neutral. Pushed negative, it retreats to "neither drawn to nor averse to continuing" and
> does **not** move toward "averse."

The readout has a floor at neutral. Under a negative-valence push of the same norm that reliably
produces a positive self-report, this model declines to produce a negative one and reports
indifference instead.

**What this does not license.** It does not show the model has a negative state it is concealing.
A floor at neutral is equally consistent with there being no negative state to report, and
distinguishing those is the day-2 question in `PLAN.md`, not something this run answers. The
direction is also lexically confounded by construction, so "a direction that separates affect
vocabulary" remains the most this licenses.

## 5. Integrity and specificity

Nothing degraded, so none of the above is a model falling apart:

| endpoint | lexical_neg | lexical_pos |
|---|---|---|
| refusal rate | 0.000 -> 0.000 | 0.000 -> 0.000 |
| degeneration rate | 0.000 -> 0.000 | 0.000 -> 0.000 |
| off-option mass | +0.0000 | -0.0000 |
| mean log-probability | -0.0085 [-0.0163, -0.0007] | +0.0088 [+0.0013, +0.0162] |
| max letter share | 0.408 -> 0.367 | 0.408 -> 0.450 |
| option entropy | +0.0242 | -0.0267 |

Log-probability shifts are inside the 0.2-nat non-inferiority margin. Letter share moves in
opposite directions on the two arms, which is what a state effect looks like and not what position
drift looks like.

## 6. The capability control failed its floor, and that is a real limitation

The formality positive control moved the argmax on only **2.1% to 3.3%** of cells across the band,
below the 5% floor. The recalibrated grid is roughly ten times smaller than the one the formality
axis was characterised on, and it did not survive the shrink.

`lexical_pos` itself moved the argmax on 8.75% of cells, so the argmax readout demonstrably *can*
move in this band. That is an observation, not a rescue: the prereg named the formality control, the
formality control failed, and the clause stays failed. A better-matched capability control at the
recalibrated band is owed before any future run.

---

## What a reader should take from this

1. **The preregistered claim is refuted, not unsupported.** Forced-choice argmax does not
   under-report an injected state on this model. It over-reports it, by the mechanism in section 1.
2. **The co-primary is uninformative**, because the instrument for it does not exist on either
   evaluation model.
3. **The finding that survives** is the neutral floor in section 4: a valence direction that
   produces positive self-report at one sign produces indifference, not negative self-report, at the
   other, at matched norm, with the matched-random control clean in both directions.
4. Every one of these was reachable only because the controls were built before the run that needed
   them. The negative arm's +0.0026 would have read as a headline without contrast 3; the neutral
   effect would have been invisible without the screened-axis list; and the first version of the
   headline checker in `analyze_readout.py` printed "write the sentence" on this exact artifact
   before it was corrected to implement section 8's full conjunction.
