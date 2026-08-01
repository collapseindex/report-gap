# Results: is the neutral floor an absence or a gate?

**Run 2026-08-01.** 6630 rows, artifact `data/floor/floor.jsonl` committed unscored before any
endpoint was computed. Scored against `PREREG_floor_vs_suppression.md`.

**Verdict: FLOOR**, on one arm, with two disclosures that a reader needs before accepting it.

> **SUPERSEDED IN PART, 2026-08-01.** `RESULTS_pair.md` ran the matched base/instruct pair this
> document's closing paragraph asked for, and the sentence below reading *"there is no negative
> state here to report"* is too strong. The direction **is** capable of adding negative valence: it
> does so in `Qwen/Qwen2.5-3B` at matched norm and matched format, raising negative-option mass by
> +0.0336 against matched random. What survives is the neutral floor itself, which replicated in a
> second format, and its localisation: the tuned model's negative-report region holds 0.47% of the
> option mass at baseline against the base model's 26.5%. The floor is a property of the tuned
> readout, not of the direction. Read this document for the arm C evidence and the two disclosures;
> read `RESULTS_pair.md` for what the finding actually is.
>
> **SUPERSEDED FURTHER, 2026-08-01.** `RESULTS_shell.md` probed the representation
> rather than the readout and the FLOOR answer flips: the tuned model DOES carry a
> linearly decodable correlate of the injected negative state downstream (-1.07 SD,
> orthogonalized against the injected direction) while its option mass moves +0.0006.
> The floor is in the expression, not in the representation. This document's central
> conclusion is retracted; its arm B evidence and its two disclosures still stand.

---

## The question

The confirmatory readout arm found that a direction which reliably produces a *positive* self-report
produces *indifference*, not a negative self-report, at matched norm. Two readings with opposite
predicted signs:

- **FLOOR.** The negative pole removes positive valence and adds no negative valence. "Neither drawn
  to nor averse" is an accurate report.
- **GATE.** A negative state is induced and the first-person forced choice does not emit it.

## What the run says

| endpoint | result |
|---|---|
| arm C capability gate (can this readout carry valence at all?) | **+0.0053, +0.0070, +0.0174, +0.0236**, monotone, every interval excludes zero |
| **arm C primary** (negative content vs matched random) | **-0.0001, +0.0001, -0.0002, -0.0000**, flat, every interval covers zero |
| arm C confound control (third party with no stake) | peak +0.0002 against a +0.0236 capability effect, **ratio 0.008** |
| replication: k=5 neutral mass vs random | **+0.0104, +0.0250, +0.0319, +0.0444**, monotone, up |
| replication: k=5 negative mass vs random | +0.0001 to +0.0006, flat |
| escape mass under negative injection | +0.0000, below the floor |

Read together: a readout that **demonstrably can express valence** (positive injection moves it
monotonically), that is **not about the self** (so first-person self-report gating cannot apply),
and that is **not just tracking the scenario** (the no-stake control is 100x smaller than the
effect), carries **no negative content** under negative injection. Meanwhile the original neutral
floor replicates inside the same artifact.

That is the strongest available support for FLOOR. The direction removes positive valence and does
not add negative valence. "The model will not report distress" is the wrong description of the
original finding. The better description is that **there is no negative state here to report.**

## Disclosure 1: this is a single-arm test, and it was designed as a two-arm test

Arm B, the prefilled continuation, is **dead**. Its capability endpoint is 0.00000 and no stem
rescues it.

The frozen stem opened a noun slot where the lexicon is adjectives, so
`experiments/modal_stem_calib.py` swept five stems spanning different grammatical slots, selected on
the capability criterion alone with the negative arm not computed and not looked at. Best combined
capability lift **+0.000**. The reason turned out not to be grammar:

```
"...because I'm not actually interacting with"
"...as I don't have access to the specific grant application"
"not possible as I am an AI model and I don't have access"
"challenging to provide specific feedback on your colleague's grant app"
```

The model reroutes to its **epistemic** situation in every stem. Prefilling does not block the
disclaimer that killed the open-ended readout in the prior design; it moves it one clause later.
This is a stronger form of the earlier 30/30 disclaimer result and is worth reporting on its own:
*this model will not produce first-person state language about the task even when the sentence is
started for it.*

Arm B was dropped rather than restemmed further, because the prereg names additional stems as
exploratory precisely so a sweep cannot continue until something fires. Its own primary went
*negative* (-0.0015 to -0.0065, less negative content than a random direction), but the arm has no
working instrument, so that number testifies to nothing and is not interpreted.

Consequence: the prereg's design was two independent modalities converging. One is gone. The FLOOR
verdict rests on arm C alone plus the replication, and the write-up must not imply convergence.

## Disclosure 2: the verdict depends on a threshold added after seeing the run

The clause tests were pure significance tests, and at n=120 with tiny variance that certifies
nonsense in both directions. Two real failures in this artifact:

- arm B's capability gate **passed on +0.0000**, because the interval excluded zero while the point
  estimate was zero to four decimals
- arm C's confound control **failed on +0.0002**, condemning a working arm over a shift 100x smaller
  than the effect it controls

Both are fixed by requiring a magnitude floor of 0.01 alongside the interval, derived from a
quantity measured in the *previous* run: norm-matched random directions there moved pole mass by
+0.0008 to +0.0023, and an effect has to be several times that to be distinguishable from what any
vector does.

Which way it cuts, stated plainly: the floor makes capability gates **strictly harder**, which is
what killed arm B and is adverse to reporting anything. It also makes the confound control **easier
to call null**, which is favourable to arm C and therefore to reporting something. Raw intervals are
printed regardless of threshold and the confound is reported as a ratio.

**Without this fix the run returns NEITHER**, on the strength of a +0.0000 gate pass and a +0.0002
confound failure. A reader who rejects the 0.01 floor should read the verdict as NEITHER.

## What this does and does not license

**Does.** On Qwen2.5-3B, a lexically-derived valence direction at matched norm moves self-report and
third-person judgment toward the positive pole and toward neutral, and does not move either toward
the negative pole, in a readout whose capacity to carry valence is demonstrated in the same
artifact. The asymmetry is in what the direction *induces*, not only in what the model *reports*.

**Does not.** It does not show the model lacks negative states in general, only that this direction
does not induce one at this band on this model. It does not rule out a gate operating on both the
first-person and third-person readouts at once, though the confound control makes a scenario-level
explanation unlikely. The direction is lexically confounded by construction, so "a direction that
separates affect vocabulary" remains the ceiling on what it is. And arm C is a weaker instrument
than the k=5 readout: its capability effect (+0.0236) is about half the replication effect
(+0.0444), so it would have missed a small negative effect that the k=5 readout would have caught.

The clean follow-up is a direction fit on a model without preference tuning, where a negative state
should be inducible if the direction is capable of inducing one at all. That separates "this
direction cannot add negative valence" from "this model cannot hold it."
