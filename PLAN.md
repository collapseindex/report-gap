# Plan

Sprint opens 2026-08-13, twelve days out. 3 days inside the window, solo.

Last updated 2026-08-01, after five preregistered arms were completed pre-window.

## The line between prep and the window

Apart confirmed that using existing infrastructure is fine, so the tooling half of this is settled:
the stimuli, hooks, scorers, controls and analysis primitives are all fair to bring. What remains is
the results half, and this repo originally held the line that every experiment would run inside the
window with `data/sweeps/` empty until it opened.

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
- *All five confirmatory arms.* The readout gap, floor-versus-gate, the base/instruct pair, the
  depth sweep and the shell/core probe, all run 2026-08-01, twelve days before the window. **These
  are results, brought rather than made.** Any sprint submission has to say so plainly and date
  them. The alternative reading, that pre-window instrument work licenses a pre-window result
  because the same repo produced both, is exactly the elision this section exists to prevent.

What this costs: the sprint contribution is no longer "here is a result produced in 3 days." What it
buys: a preregistered chain where each arm answers the objection the last one raised, including one
raised by the literature and one that overturned our own conclusion. That chain could not have been
built in 3 days, because each link had to be preregistered before the next was designed. The honest
framing is that this is completed work being brought to a sprint, and the sprint days are for the
two extensions listed below that would genuinely widen it.

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

Rewritten 2026-08-01. Both extensions this section previously listed for days 1 and 2 have since
been run, twelve days early, and are disclosed as such at the top of this file.

**Already done, pre-window:** the base/instruct pair (`RESULTS_pair.md`), the depth sweep answering
the literature's strongest objection (`RESULTS_depth.md`), and the representation probe that
overturned our own FLOOR conclusion (`RESULTS_shell.md`).

What is genuinely left, in order of what a null would teach:

- **Day 1. A second architecture family that responds at all.** Every positive result rests on
  Qwen. Llama is inert on this readout at both sizes tested, on live readouts with plenty of
  headroom. Gemma-2 or Mistral would tell us whether the neutral floor is a property of preference
  tuning in general or of one family's tuning. This is the single biggest limitation and the
  cheapest to attack.

  Note for anyone proposing "refit the direction on Llama instead of porting it": the direction has
  **always** been fit per model, at the injection layer, never transferred. `modal_alpha_recal.py`,
  `modal_base_pair.py`, `modal_depth.py` and `modal_shell_core.py` all call `fit_direction` on the
  model under test. Cross-model transfer is named exploratory in every prereg and has not been run.
  Llama's inertness is with its own fitted direction, which is what makes it a finding rather than
  a porting failure.

- **Day 1b. Induce the state by a route that is not this direction.** The sharpest objection to the
  SHELL result is that orthogonalizing the probe removes the injected vector but not directions
  correlated with it, so the probe may be reading the wake of the push rather than a carried state.
  A prompt that genuinely makes the task aversive, with no injection at all, would separate them: if
  the same probe fires and the same options stay flat, the claim is about the model rather than
  about our intervention. This is cheap and it is the single thing that would most strengthen the
  headline.
- **Day 2. A non-lexical direction.** The surviving direction is fit from first-person state
  language and is lexically confounded by construction; the task axis built to avoid that failed its
  decoding gate at three scales. Without this, "a direction that separates affect vocabulary"
  remains the ceiling on what the whole project is about. Options: contrastive prompts holding
  vocabulary fixed, or an SAE feature selected for valence without lexical supervision.
- **Day 3.** Figures, write-up, submission, with the pre-window dating stated in the submission
  rather than in a footnote. `run_all.py` from `paper-harness` in the loop.

If the second family is also inert, that is a better result than another Qwen number: it would mean
this method of inducing a self-reportable state does not generalize, which the field should know
before building on it.

## Risks, ranked

Reordered after the pre-window runs. Risks 1 and 3 have already fired.

1. **FIRED. The direction has no consequence at some strengths on some models.** Llama-3.1-8B is
   inert entirely and the negative pole is inert on Qwen2.5-3B. The positive control and the
   responsiveness rule are what separated "the instrument is at fault" from "the direction is," and
   both were in place before the runs that needed them.
2. **FIRED, and survived. The depth objection.** The literature puts negative valence at a depth we
   never injected at. Tested across eight depths; the null held at all seven gate-clean ones.
   Recorded because a risk that fires and survives is worth as much as one that never fires.
3. **FIRED, in an unexpected form. Saturation, not coherence collapse.** The expected failure was the
   model degenerating at high alpha. What actually happened is the readout pinning while every
   coherence endpoint stayed clean, which the original band check could not see. Fixed with the
   saturation and liveness criteria.
4. **Bringing a result to a sprint that is for making them.** The live risk, and it grew: five arms
   are now complete pre-window, not one. See the section at the top of this file. The mitigation is
   disclosure, not framing.
5. **One architecture family carries every positive result.** The sharpest remaining limitation.
   Llama is inert at two sizes on live readouts, so "the neutral floor" could be a Qwen fact rather
   than a tuning fact. Day 1 above exists for this.
6. **The direction is lexically confounded and the non-lexical alternative failed.** Conceded in
   every prereg from the start. It caps what the entire project is entitled to say, and no amount
   of downstream rigour lifts that cap. Day 2 above is the only thing that would.
7. **We overturned our own conclusion once.** FLOOR became SHELL when we probed the representation
   instead of the readout. That is the process working, but it is also a warning that the current
   headline is one good follow-up away from moving again, and the write-up should not read as
   settled.
