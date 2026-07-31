# Plan

Sprint opens in 13 days. 3 days inside the window, solo.

## The line between prep and the window

Apart sprints allow you to bring tooling. They do not allow you to bring results. The line this repo
holds:

**Built before the window:** reusable infrastructure, the frozen preregistration, stimuli, the
injection hooks, the judge-free scorers, and every unit test and negative test. All of it is
instrument construction, and the prereg's own timestamp check proves it landed before any artifact
did.

**Run inside the window:** every experiment. `results/` and `data/sweeps/` stay empty until the
sprint opens. This is not a technicality, it is what makes the prereg mean anything: `check_prereg.py`
compares the commit that added the prereg against the commit that added the first results artifact,
and a prereg that landed second is a diary entry.

Sprint-day work is then the interesting half, because the boring half cannot fail at 2am.

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

- **Day 1.** Positive control and confabulation floor first, before the treatment arm. If self-report
  cannot move, or the floor is at ceiling, the design needs to change and that is worth knowing on
  day 1 rather than day 3. Then the alpha = 0 baseline and the unhooked placebo.
- **Day 2.** Full treatment sweep on Qwen2.5-3B, then the norm-matched random control on the same
  grid. Replication on Llama-3.1-8B if the budget holds. Raw generations committed before any scoring
  runs.
- **Day 3.** Scoring, intervals, the gap map figure, write-up. `run_all.py` from `paper-harness` in
  the loop. Submit.

## Risks, ranked

1. **The direction has no behavioral consequence at any strength.** The most likely way this returns
   nothing. Mitigated by the positive control, which tells you whether the instrument or the
   direction is at fault, and by the interpretation table, which already says what each of those two
   nulls licenses.
2. **Trivial decodability makes R1 uninteresting.** Already conceded in the prereg. R1 is descriptive
   and the headline contrast is between two behavioral outputs.
3. **Coherence collapse eats the high-alpha cells.** Expected. The exclusion threshold is frozen and
   the excluded cells get reported rather than dropped.
4. **13 days of prep with no run means a day-1 surprise.** The n=2 CPU selftest is the whole defense.
   If it runs end to end on two items, the only thing the window adds is scale.
