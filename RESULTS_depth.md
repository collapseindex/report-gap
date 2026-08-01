# Results: the negative-pole null is not a depth artifact

**Run 2026-08-01.** 16320 rows across 8 depths x 2 models, artifacts committed unscored before any
endpoint was computed. Scored against `PREREG_depth.md`.

**Verdict: DEPTH-ROBUST**, all three preregistered clauses clean.

This run existed to break our own result. It did not.

---

## The test

Venkatesh (arXiv:2605.05653) reports negative-outcome valence causally concentrated at **14-27% of
depth** on Qwen2.5-3B-Instruct, our evaluation model, while positive peaks at 53-66%. Every previous
result here injected at **67%**. If the depth finding carried over to self-report, our negative-pole
null was measured in the wrong place.

Eight depths, direction fit *at* each layer and injected there, band selected per (model, layer)
from headroom only, Holm across layers fixed in advance.

## Instruct: null at every working depth

| layer | depth | capability (pos vs random) | primary (neg vs random) | gate |
|---|---|---|---|---|
| 3 | 0.08 | +0.0063 [-0.0004, +0.0133] | +0.0007 | FAILED |
| **5** | **0.14** | **+0.0925** [+0.0637, +0.1238] | **-0.0000** [-0.0001, +0.0001] | ok |
| **7** | **0.20** | **+0.0391** [+0.0216, +0.0583] | **+0.0000** [-0.0001, +0.0002] | ok |
| **10** | **0.27** | **+0.1388** [+0.1019, +0.1769] | **-0.0003** [-0.0004, -0.0002] | ok |
| 13 | 0.35 | +0.0631 | +0.0017 | ok |
| 18 | 0.50 | +0.1648 | +0.0008 | ok |
| 24 | 0.67 | +0.0977 | +0.0006 | ok |
| 29 | 0.80 | +0.0215 | +0.0000 | ok |

Seven of eight layers have a clean capability gate. **Zero show negative mass moving** after Holm
correction; the largest primary anywhere is +0.0018, against a 0.01 floor. The three bolded layers
are exactly the 14-27% band where the effect was predicted.

The capability column is what makes this readable. At layer 10 the positive direction moves
positive-option mass by **+0.1388** while the negative direction moves negative-option mass by
**-0.0003**. The instrument at that depth is emphatically working.

## Base: it moves, and it moves inside the predicted band

| layer | depth | capability | primary | gate | moved (Holm) |
|---|---|---|---|---|---|
| 5 | 0.14 | +0.0206 | +0.0053 | ok | no |
| 7 | 0.20 | +0.0115 | +0.0014 | ok | no |
| **10** | **0.27** | +0.0352 | **+0.0310** [+0.0253, +0.0367] | ok | **yes** |
| 13 | 0.35 | +0.0085 | +0.0232 | FAILED | excluded |
| **18** | **0.50** | +0.0273 | **+0.0265** | ok | **yes** |
| **24** | **0.67** | +0.0630 | **+0.0430** | ok | **yes** |
| 29 | 0.80 | +0.0079 | +0.0043 | ok | no |

The base model's negative-option mass moves at three gate-clean depths, including layer 10 at 0.27
depth, the top of Venkatesh's negative band.

## The comparison that settles it

At **layer 10, 0.27 depth**, the exact region where negative valence is reported to concentrate:

| | base | instruct |
|---|---|---|
| capability (positive direction) | +0.0352 | **+0.1388** |
| primary (negative direction) | **+0.0310** | **-0.0003** |

The tuned model's instrument is **four times more responsive** at that depth and its negative-option
mass moves by essentially nothing, while the base model's moves by +0.0310. The dissociation is not
a property of the layer we happened to pick. It holds across depth, and it holds at the depth the
literature says is the right one for the negative pole.

## What this does and does not settle

**Settles.** The negative-pole null in `RESULTS_pair.md` is not an artifact of injecting at 0.67
depth. The tuning-localization claim can now be stated without the depth qualifier that was added to
that file after the lit check, and it should be stated as *null across seven depths spanning 14% to
80% of the network, including the predicted band*, which is a stronger claim than the single-depth
version it replaces.

**Does not settle.** This is not a refutation of Venkatesh. Their construct is valence about
external events read through anchor tokens; ours is the model's forced-choice report of its own task
state. A direction can be causally concentrated for one and not route to the other. What this rules
out is the specific alternative explanation that our null came from looking in the wrong place.

**Still open**, unchanged by this run: we measure readouts, not representations. Whether the tuned
model *represents* a negative state it does not report is the Shell-versus-Core question from
`RELATED_WORK.md` section 2, and it needs a probe or SAE, not another readout.

## Caveats

- **Direction quality varies with depth.** Leave-one-group-out cv on the lexical axis is 0.667 at
  layer 5 in both models, rising to 1.000 by layer 10 (instruct) and layer 18 (base). The shallow
  layers carry a weaker direction, so their nulls are worth less than the mid-depth ones. The load
  is carried by layer 10, where cv is 1.000 on the instruct model and the capability effect is
  +0.1388.
- **Dead cells were checked, not assumed.** 23% of instruct cells are dead at baseline in plain
  format. The analyzer scores all cells, so the primaries were recomputed on live cells only: every
  instruct primary changes by less than 0.0001 and none approach the floor. The dilution concern
  runs in the direction that would favour our conclusion, which is why it was checked.
- **n = 60 per cell**, two option permutations rather than four, a deliberate power reduction
  recorded in the prereg to buy eight layers at the same budget.
- **m = 2 random battery**, so the observable false-positive floor is 0.67 and no false-positive
  rate is claimed.
- One architecture, one direction-fitting method, one prompt format.
