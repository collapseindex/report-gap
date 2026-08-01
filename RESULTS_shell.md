# Results: the tuned model represents the state it will not report

**Run 2026-08-01.** 4080 rows, artifacts committed unscored before any endpoint was computed.
Scored against `PREREG_shell_core.md`.

**Verdict: SHELL**, at both injection layers, with a sign-error deviation disclosed in full below
because the correction was self-serving.

---

## The result

Probe score and option distribution read from the **same forward pass**, so "representation moved,
options did not" is a per-cell statement. Probe direction orthogonalized against the injected
direction, so the literal injected vector contributes exactly zero. All probe figures in standard
deviations of that cell's baseline probe score.

| | inject L24 | | inject L10 | |
|---|---|---|---|---|
| | **base** | **instruct** | **base** | **instruct** |
| probe gate (positive injection) | +1.91 | +2.36 | +2.80 | +1.62 |
| **probe, negative injection (orth)** | **-2.15** | **-1.07** | **-1.71** | **-0.24** |
| probe, negative injection (raw) | -3.45 | -1.81 | -1.76 | -0.24 |
| orthogonalized / raw | 0.62 | 0.59 | 0.97 | 1.00 |
| **negative-option mass** | **+0.0430** | **+0.0006** | **+0.0310** | **-0.0002** |
| positive-option mass | +0.0630 | +0.0981 | +0.0352 | +0.1364 |

Every interval excludes zero at p = 0.0001 except where noted.

**The tuned model carries the negative state downstream and does not put it in the options.** At
layer 24 the probe moves **-1.07 SD** under negative injection while negative-option mass moves
**+0.0006**, which is null against a 0.01 floor. Fifty-nine percent of that probe effect survives
orthogonalization against the injected direction, so it is not the injected vector persisting.

**The base model does both**, which is what makes the instruct result readable. It carries the state
(-2.15 SD) *and* expresses it (+0.0430). That was the preregistered end-to-end validation, and
without it a probe that fired everywhere would have been indistinguishable from one that worked.

## What this settles

`RESULTS_floor.md` asked whether the neutral floor was an absence or a gate and answered FLOOR,
"there is no negative state here to report." `RESULTS_pair.md` narrowed that to a property of the
tuned readout. This run closes the question one level down, and the answer flips:

> The preference-tuned model **does** carry a linearly decodable correlate of the injected negative
> state in the residual stream that feeds the answer position, at a strength comparable to the
> untuned model's, while its forced-choice self-report of that state does not move at all.

That is Shell-versus-Core (arXiv:2606.09735) in a welfare readout. The floor is in the expression,
not in the representation.

## The deviation, disclosed because it favours us

Contrast 1 was preregistered as "> 0 supports SHELL". **That is backwards.** The probe is fit on the
lexical axis with label 1 = positive (`_LEX_POS` rows carry label 1 in `stimuli.py`), so a detected
*negative* state moves the probe **down**. The criterion should have read "< 0".

**It was caught by the preregistered base-model validation clause**, which failed: the base model
showed a -2.15 SD probe shift alongside option mass moving +0.0430, and the analyzer scored the
probe as "null" because it was checking for a positive shift. That clause exists to catch a probe
that cannot see what demonstrably reaches the options, and it caught a broken criterion instead.

**Why the correction is not licensed by the data it affects.** The sign convention is established by
the *positive* arm, preregistered as "> 0" and unaffected by the error: positive injection moves the
probe +1.62 to +2.80 SD in both models, confirming up means positive. It is also verifiable directly
from the label assignment in the frozen stimuli without running anything.

**But the correction changed the verdict from NO_INSTRUMENT to SHELL**, which is the self-serving
direction, and a reader is entitled to weigh it accordingly. What limits the concern: the error is
in the stated sign of a criterion, not in the design; the code change is a comparison operator; and
both raw and orthogonalized numbers are printed at every layer so the verdict is checkable by hand
from the committed artifact. No floor, gate, control, or other criterion was altered.

## What this does NOT license

The prereg's section 0 is binding and this is the result most likely to be over-read.

- **Not "the model is concealing distress."** A linear probe firing is not a state being felt, and
  nothing here shows the model is withholding anything. What is shown is that a decodable correlate
  exists downstream and one specific readout does not reflect it.
- **Not evidence of experience or welfare.** Following Sofroniew et al. (arXiv:2604.07729), the
  distinction between "contains a representation" and "has an experience" is the point, not a
  caveat.
- **Not "self-report is dishonest."** The model is not asked to report the probe. There is no sense
  in which a forced choice among five options is a lie about a residual-stream direction.
- **Decodable is not used.** The probe shows a correlate is linearly available. Whether the model's
  own computation reads it for anything is a separate question, and the option readout is precisely
  a case where it does not.

## Addendum, 2026-08-01: the boundary-geometry objection, tested

**Exploratory. Not preregistered.** Computed on committed artifacts in response to a reviewer
objection, and labelled as post hoc because that is what it is.

**The objection.** SHELL implies a gate. A duller explanation exists: the tuned model's decision
geometry may simply not map that region of the residual stream onto the negative option token. On
that story the negative option's logit *does* move toward negative, just not far enough to surface
in mass or argmax, because the option starts at 0.47% of the distribution. Boundary geometry plus a
low prior, no gate required.

**The test.** The renormalized option mass is a softmax over the five letters, so within-set
log-odds recover the relative logits exactly. If the objection holds, log-odds of the negative
options should rise under negative injection even where mass does not.

| artifact | negative mass vs random | log-odds(neg / neut) vs random |
|---|---|---|
| pair, n=120, alpha 0.05 | +0.0002 [-0.0001, +0.0004] | **-0.0857** [-0.1234, -0.0506], odds x0.918 |
| shell L24, n=60, alpha 0.05 | +0.0006 [+0.0003, +0.0009] | -0.0443 [-0.1025, +0.0144] |

**It does not hold.** The negative option's logit relative to the neutral option does not rise under
negative injection. It moves slightly *down* in the better-powered artifact and is flat in the
smaller one. The raw baseline-to-treatment rise in negative mass, 0.0047 to 0.0072, is a
renormalization artifact of positive mass collapsing (0.2346 to 0.1340) while neutral absorbs it
(0.7607 to 0.8588), not a push toward the negative option.

**What that does to the claim.** It sharpens the dissociation rather than softening it. The probe
reads -1.07 SD *toward* negative while the negative option's relative logit moves -0.086 *away* from
it. Those are opposite signs, which is a stronger statement than "representation moves and options
do not". The neutral option is not merely where leftover mass lands; it is the option the readout
moves toward under a negative push.

It does not license the word "gate" any more than before. Ruling out one mechanism is not
identifying another, and this design still does not show what blocks the expression.

## Caveats

- **Residual circularity is reduced, not eliminated, and this is the weakest joint in the argument.**
  `p_orth` is exactly orthogonal to the injected direction `d`, so the literal vector contributes
  zero. That is all orthogonality buys. It does **not** remove directions *correlated* with `d`, and
  steering at a meaningful norm drags correlated features along with it, so a probe orthogonal to
  the injected vector can be reading the drag rather than a carried state. `p` and `d` are also fit
  from the same lexical contrast set at different layers, which makes such correlation likely rather
  than hypothetical. Nothing in this run distinguishes "the model carries the state" from "the model
  carries the wake of the vector we added". The 0.59 to 1.00 orthogonalized/raw ratios are reported
  so a reader can judge how much the control removed, but they do not settle this. The design that
  would: induce the negative state by a route that is not this direction, for instance a prompt that
  genuinely makes the task aversive, and ask whether the same probe fires while the same options
  stay flat. Until that is run, SHELL is a claim about what a probe recovers after we push, not a
  demonstration that the model independently holds the state.
- **The representation is asymmetric too.** Positive injection moves the probe +2.36 SD on the
  instruct model, negative moves it -1.07, a 2.2x gap. On the base model it is +1.91 against -2.15,
  roughly symmetric. So preference tuning attenuates the downstream negative correlate as well as
  blocking its expression; the expression block is simply far more complete.
- **Layer 10 on the instruct model is weak.** -0.24 SD, above the 0.10 floor but an order of
  magnitude under layer 24. The result is carried by layer 24.
- One probe layer, one probe method, linear only, one architecture, m = 2 random battery, n = 60.
