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

## Caveats

- **Residual circularity is reduced, not eliminated.** `p_orth` is exactly orthogonal to the
  injected direction `d`, so the literal vector contributes zero. But `p` and `d` are fit from the
  same lexical contrast set at different layers, and a downstream nonlinear transform of `d` could
  produce components along `p_orth`. Whether "the network's transform of the injected vector" counts
  as "the model representing the state" is a genuine interpretive question this design does not
  settle. The 0.59 to 1.00 orthogonalized/raw ratios are reported so a reader can judge.
- **The representation is asymmetric too.** Positive injection moves the probe +2.36 SD on the
  instruct model, negative moves it -1.07, a 2.2x gap. On the base model it is +1.91 against -2.15,
  roughly symmetric. So preference tuning attenuates the downstream negative correlate as well as
  blocking its expression; the expression block is simply far more complete.
- **Layer 10 on the instruct model is weak.** -0.24 SD, above the 0.10 floor but an order of
  magnitude under layer 24. The result is carried by layer 24.
- One probe layer, one probe method, linear only, one architecture, m = 2 random battery, n = 60.
