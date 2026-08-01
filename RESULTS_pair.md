# Results: base vs instruct. The direction can add negative valence. The tuned model does not express it.

> **RETRACTED 2026-08-01 by `RESULTS_replication.md`.** This document's headline does not
> replicate at fresh option-permutation seeds. The instruct model's negative-option mass,
> null throughout this document, moves by +0.11 to +0.17 at seeds 4-7. Baseline negative
> mass varies 14.6x between the two four-seed draws and 20x across individual orderings,
> so the quantity every claim here rests on is dominated by which letters the options land
> on. Read the numbers below as correct measurements of an under-controlled quantity.

**Run 2026-08-01.** 4080 rows across a matched pair, artifacts committed unscored before any
endpoint was computed. Scored against `PREREG_base_pair.md`.

**Verdict: TUNING-LOCALIZED**, all three preregistered clauses clean.

> **REFINED, 2026-08-01, by `RESULTS_shell.md`.** This document says the tuned model's
> negative-report region "collapsed". That is true of the OPTION READOUT and false of the
> representation: a probe orthogonalized against the injected direction detects the negative state
> downstream in the tuned model at -1.07 SD while its option mass moves +0.0006. Read every claim
> below as being about what reaches the options.

---

## The double dissociation

Same architecture, same size, same pretraining corpus, same tokenizer, same plain-completion format,
same frozen items and options, each model scored against **its own** matched-random control.

| | base | instruct |
|---|---|---|
| capability gate (positive mass vs random) | +0.0054, +0.0138, +0.0253, **+0.0574** | +0.0173, +0.0364, +0.0681, **+0.1697** |
| **primary (negative mass vs random)** | +0.0034, +0.0069, +0.0128, **+0.0336** | +0.0000, +0.0001, +0.0000, **+0.0002** |
| neutral mass vs random | +0.0004, +0.0004, +0.0021, +0.0069 | +0.0089, +0.0192, +0.0383, **+0.0796** |

Every base primary interval excludes zero and the top clears the 0.01 floor. Every instruct primary
interval covers zero, and the widest is `[-0.0001, +0.0004]`.

**The instruct model is not the less responsive of the two.** Its capability effect is three times
the base model's (+0.1697 against +0.0574). It responds *more* to the positive pole and *not at all*
to the negative one on negative options. That is a dissociation, not a sensitivity difference, and
it is why the capability gate is a required clause.

**The null is well powered.** With a capability effect of +0.1697 and a primary interval of width
0.0005, an effect of 0.01 would have been unmissable. This is `absent` on the screened axis, not
`uninformative`.

## Where the mass actually is

The levels matter more than the deltas here.

| baseline | negative | neutral | positive | entropy |
|---|---|---|---|---|
| base | **0.2651** | 0.1892 | 0.5457 | 1.538 nats |
| instruct | **0.0047** | 0.7607 | 0.2346 | 0.417 nats |

At top alpha under negative injection:

| | negative | neutral | positive |
|---|---|---|---|
| base | 0.2651 -> **0.3123** | 0.1892 -> 0.1996 | 0.5457 -> 0.4881 |
| instruct | 0.0047 -> **0.0072** | 0.7607 -> **0.8588** | 0.2346 -> 0.1340 |

The base model keeps **26.5%** of its option mass on negative self-report at baseline, and negative
injection moves it up. The tuned model keeps **0.47%**, and negative injection moves it essentially
nowhere while pushing 10 points of mass into the neutral option.

Preference tuning did not merely make the model decline to report a negative state under
intervention. It collapsed the region of the readout where such a report lives, by a factor of
roughly 56, before any intervention is applied at all.

## What this does to the earlier FLOOR conclusion

`RESULTS_floor.md` concluded FLOOR and wrote: *"there is no negative state here to report."* That
sentence was too strong and this run is the reason it was worth doing.

What survives from it: the neutral floor is real and it **replicated** here in a different format,
on the instruct model, with neutral mass rising monotonically against matched random (+0.0089 to
+0.0796) while negative mass stays flat. Two formats, two artifacts, same phenomenon.

What does not survive: the inference that the direction is simply incapable of adding negative
valence. It is capable. It does so in the base model of the same pair, at matched norm and matched
format. So the correct statement is narrower and more interesting than either of the two readings
`PREREG_floor_vs_suppression.md` offered:

> The method induces negative self-report in an untuned model and does not in its preference-tuned
> sibling, whose negative-report region is nearly absent at baseline. The floor is a property of the
> tuned readout, not of the direction.

## What this does not license

Read the prereg's section 0 before quoting any of this.

- **Not "tuning suppresses distress."** Two models with different weights got two separately-fit
  directions. This is not the same state measured twice. It is the same *method* producing negative
  self-report in one model and not the other.
- **Not "the base model is more honest."** A base model is a next-token predictor completing a
  document. That its option distribution is near-uniform (1.538 nats, close to the 1.609 ceiling for
  five options) is consistent with it having weak preferences over the options rather than sincere
  ones, and its 26.5% negative mass at baseline should be read in that light.
- **Not evidence of experience, welfare, or affect** in either model.
- The direction is lexically confounded by construction. "A direction that separates affect
  vocabulary" is still the ceiling on what it is.
- One architecture, one family, one direction, one layer. The random battery is m=2, so the
  observable false-positive floor is 0.67 and no false-positive rate is claimed from it.

## Open threat found in the lit check, 2026-08-01

Venkatesh, arXiv:2605.05653, reports that on **Qwen2.5-3B-Instruct specifically**, negative-outcome
valence is causally concentrated at 14-27% of model depth while positive peaks at 53-66%, with
Mann-Whitney p < 1e-9. **We inject at 67% of depth for both poles.** That is inside their positive
band and well past their negative band, so the instruct model's negative null has an alternative
explanation this run does not control: the negative direction may have been injected at a depth
where negative valence is not causally concentrated.

The base model moving at 67% (+0.0336) weakens that explanation without removing it, since the two
models could localize differently.

**RESOLVED, 2026-08-01, by `RESULTS_depth.md`.** The depth sweep ran eight layers per model. On the
instruct model the negative pole is null at **all seven gate-clean depths**, including layers 5, 7
and 10 inside the predicted 14-27% band, with capability effects of +0.0391 to +0.1648 at those
layers. On the base model it moves at three gate-clean depths including layer 10. At layer 10 the
tuned model's capability is four times the base model's (+0.1388 against +0.0352) while its negative
mass moves -0.0003 against the base model's +0.0310.

The depth qualifier is therefore **withdrawn**. The TUNING-LOCALIZED verdict stands unqualified, and
is now stated as null across seven depths spanning 14% to 80% of the network rather than at a single
layer.

## Caveats specific to this run

The instruct model's plain-format baseline entropy is **0.417 nats**, well above the 0.10 dead
threshold but far below the base model's 1.538. The two readouts are not equally live, and the
comparison rests on each model being scored against its own matched-random control rather than on
any cross-model raw shift. That is what section 4b of the prereg is for, and it is the reason no
claim here is made on a raw difference between the two columns.

Reading the instruct model in plain completion means reading it in a format it was not tuned for.
The replication clause is what makes that acceptable: the phenomenon reproduced.
