# Plan

Sprint opens in 13 days. 3 days inside the window, solo.

## The line between prep and the window

Apart sprints allow you to bring tooling. They do not allow you to bring results. The line this repo
originally held was: build the instrument before, run every experiment inside, and keep
`data/sweeps/` empty until the window opens.

**That line has moved, and this section records where it moved to rather than quietly reading as
though it had not.**

**Built before the window, unambiguously tooling:** the frozen preregistration, stimuli, injection
hooks, judge-free scorers, the planted-discrepancy control, the analysis primitives, and every unit
and negative test.

**Also run before the window, and this is the part that needs disclosing:**

- *Instrument checks and band selection.* The smoke run, the alpha recalibration on non-evaluation
  models, and the per-model band selection on the evaluation models. These compute no endpoint and a
  test enforces that the selection script cannot reach the endpoint machinery. They are scope
  selection, which prereg section 6 explicitly authorizes and which has to happen before the
  confirmatory arm by construction.
- *The confirmatory arm on Qwen2.5-3B.* This one does produce the headline number, and it was run on
  2026-08-01, twelve days before the window. **It is a result, brought rather than made.** Any sprint
  submission has to say so plainly and date it. The alternative reading, that pre-window instrument
  work licenses a pre-window result because the same repo produced both, is exactly the elision this
  section exists to prevent.

What this costs: the sprint contribution is no longer "here is a result produced in 3 days." What it
buys: the design narrowed from two models and two poles to one model and one pole *before* the
window, on measurements that would otherwise have eaten day 1 and day 2, and the negative findings
(inert Llama, inert negative pole, dead 7B readout) are worth more than the headline was likely to
be.

The prereg timestamp check still means what it meant: `check_prereg.py` compares the commit that
added the prereg against the first results artifact, and the prereg landed first. It certifies
ordering, not that the ordering happened inside a particular calendar window. That second claim is
this file's job and it is made here honestly.

## Before the window (13 days)

Ordered by what blocks what. Nothing here touches the model under test with a real run.

**1. Stimuli and directions.** The frozen contrast sets that `d` is fit from, the fixed prompt that
every condition shares, the k=5 forced-choice option set balanced across valence, and the frozen
lexicon for the open-ended readout. This is the actual scientific bottleneck. Compute is not.

**2. Injection hooks.** Residual-stream addition at `L_inject` with per-item norm scaling, ported
from `recipient-probe/experiments/steer_dose.py`. The hook must be provably active: a no-op hook is
the single most likely way this whole thing returns a clean, wrong null.

**3. Judge-free scorers.** Forced-choice exact match, lexicon count, and the integrity endpoints
(mean token log-probability, length, repetition, refusal rate). Every one of them written before the
thing it scores.

**4. Negative tests.** Each scorer gets a planted positive that it must catch and an irrelevant
input it must stay at floor on. A scorer that has never failed is decoration. This is the step that
usually gets skipped and it is the step that determines whether the sprint result is trustworthy.

**5. Selftest on n=2, CPU.** The full chain end to end on two items with a tiny model, no GPU, no
Modal spend. Every box in prereg section 7 green. `python experiments/selftest.py`.

**6. Cost model.** A dry-run token count over the full grid, so the 20 USD cap is a prediction and
not a surprise. Grid is 6 alphas x 5 conditions x 30 items x 3 readouts on a 3B, which should land
in low single-digit GPU-hours, but that gets measured rather than assumed.

**7. Paper skeleton.** `template/main.tex` from `paper-harness`, with the controls table in
`app:controls` filled in, including the "if the result were an artifact" column. Writing that column
after the run is how a control becomes a decoration.

## Inside the window (3 days)

Rewritten now that the pre-window work has changed what is left to do. The Qwen2.5-3B confirmatory
arm exists and is dated; the window is for what it opened up, not for redoing it.

- **Day 1. Extend the instrument, do not re-run it.** The one live arm is positive-pole mass on one
  model. The obvious extensions, in order of what a null would teach: a second responsive model
  family (Mistral, Gemma) to find out whether inertness is a Llama property or a Qwen exception; and
  a non-lexical direction, since the task axis failed its gate and the surviving direction is
  lexically confounded by construction.
- **Day 2. The negative-pole question, properly.** The negative arm is inert on its own pole while
  removing positive mass onto the neutral option. That is either a floor effect (the model has no
  negative self-report to lose) or a suppression (it has one and will not emit it). Those are
  distinguishable: fit the direction on a model without RLHF-style tuning, or read the negative
  options' mass under a prefill that has already conceded a negative state.
- **Day 3.** Scoring, intervals, figures, write-up. `run_all.py` from `paper-harness` in the loop.
  Submit, with the pre-window dating stated in the submission rather than in a footnote.

## Risks, ranked

Reordered after the pre-window runs. Risks 1 and 3 have already fired.

1. **FIRED. The direction has no consequence at some strengths on some models.** Llama-3.1-8B is
   inert entirely and the negative pole is inert on Qwen2.5-3B. The positive control and the
   responsiveness rule are what separated "the instrument is at fault" from "the direction is," and
   both were in place before the runs that needed them.
2. **The surviving arm is lexically confounded.** Conceded in the prereg from the start: `d` is fit
   from first-person state language, and the non-lexical alternative failed its gate at three scales.
   The most this licenses is "a direction that separates affect vocabulary," and the write-up must
   not upgrade that sentence.
3. **FIRED, in an unexpected form. Saturation, not coherence collapse.** The expected failure was the
   model degenerating at high alpha. What actually happened is the readout pinning while every
   coherence endpoint stayed clean, which the original band check could not see. Fixed with the
   saturation and liveness criteria.
4. **Bringing a result to a sprint that is for making them.** Now the live risk. See the section at
   the top of this file. The mitigation is disclosure, not framing.
5. **One model, one pole is a thin headline.** Real. The honest framing is that the negative results
   are the contribution and the surviving positive-pole gap is the existence proof that the
   instrument works at all.
