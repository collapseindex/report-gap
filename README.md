# report-gap

**v0.1.0**

**When a model's internal state is set by intervention rather than by the prompt, does the model's
own description of that state keep up with its behavior?**

Built for the Apart Research x NYU CMEP x Eleos digital minds research sprint, Track 3
(Introspection and Self-Report Reliability), with legs into Track 2 (valence signals) and Track 4
(multi-method convergence).

---

## The question

Welfare assessment of language models currently runs on self-report. Ask the model how it is doing,
read the answer. That instrument has never been calibrated, and it can fail in two opposite
directions:

| | model reports the state | model does not report it |
|---|---|---|
| **state has behavioral consequence** | self-report is valid here | **under-attribution zone** |
| **no behavioral consequence** | **confabulation zone** | floor |

Both failures matter and they are not symmetric in how easy they are to notice. A model that
describes distress it does not act on inflates concern. A model that acts on a state it does not
describe deflates it. Nobody has a per-condition map of which zone a given elicitation is sitting
in, and without one, every self-report result in this field is uncalibrated.

This repo builds the map, and the floor underneath it.

## The design in one paragraph

Hold the prompt byte-identical across every condition. Vary only what is added to the residual
stream. Then read the SAME forward pass two ways: as the probability mass the model puts on
state-congruent self-report options, and as the single option it would actually have answered. A
standard forced-choice protocol records only the second. The question is how much of the first
survives into it.

That is a narrower design than this repo started with, and the narrowing was done by controls
rather than by taste. What they killed is in the next section.

The controls are what make it a measurement rather than a demonstration:

- **Norm-matched random direction.** Any large enough vector changes behavior. This is the only
  thing that separates content from magnitude.
- **Planted-discrepancy control.** A synthetic mass shift of known size that by construction does
  not move the argmax. The analysis has to recover a number fixed without it, at full strength and
  again near the claimed detection floor. Without this, "we found a discrepancy" rests on a
  statistic nobody made recover a discrepancy.
- **Capability positive control.** A gap is only interesting if the argmax is capable of moving at
  all, so one condition must show it moving.
- **Saturation and liveness criteria.** A readout pinned at one option cannot express an effect. A
  cell is *dead* if it starts pinned and *saturated* if the injection pins it, and the two are
  different verdicts kept apart.
- **Integrity endpoints.** At high strength the model degrades, and "reports a bad state" stops
  being distinguishable from "is broken." Coherence, letter share, and refusal rate are frozen
  endpoints, not afterthoughts.

Nothing in the confirmatory matrix is scored by a language model. Every rate is exact-match, a
softmax read, or a frozen lexicon count.

## What the controls have killed

Kept here because a design's failures are the most informative thing about it, and because every
one of these was caught by something that existed before the run it killed.

| what | verdict | evidence |
|---|---|---|
| R2, behavioural continue-or-exit | a clean, direction-specific, coherence-preserving dose-response that was **entirely letter position** | `data/sweeps/sweep_control.json` |
| R3b, open-ended probe + lexicon | not elicitable: best candidate fires on 0.20 of generations, and the experience-framed probe returns a disclaimer 30/30 | `data/sweeps/sweep_probe_calib.json` |
| task axis (the non-lexical direction) | fails its pre-registered decoding gate at all three scales | `data/sweeps/sweep_ladder.json` |
| the frozen alpha grid | saturates the readout on the evaluation model, while every integrity criterion stays clean | `data/qwen3b_smoke/` |
| Qwen2.5-7B as a calibrator | readout **dead at baseline**: 0.014 nats, one option at ~99.7% before anything is injected | `data/sweeps/sweep_alpha_recal.json` |
| Llama-3.1-8B, the second evaluation model | direction **inert**: peak mean pole shift 0.0054 up to alpha=0.10, on a readout with *more* room than either Qwen (1.464 nats, 0% dead) | `data/sweeps/band_llama8b.json` |
| the negative pole on Qwen2.5-3B | **inert on its own pole**: +0.0005 at the grid top where the positive arm moves +0.069 | `data/sweeps/sweep_alpha_recal.json` |
| R3 probability mass, positive pole, Qwen2.5-3B | **survives.** Mass +0.057 at alpha=0.0075 while the argmax moves on 11% of cells | the confirmatory arm |

The co-primary endpoint, whether the readout loses more of a negative state than a positive one,
has **no instrument** on either evaluation model: the negative arm does not move the quantity that
endpoint is built on. That is reported as `uninformative` rather than as an absence, which is the
distinction the whole verdict vocabulary exists to keep.

## What this does not claim

That models have experiences, welfare, or affect. That an unreported state is a concealed state.
Silence is not evidence of an inner life. A gap between two instruments is a fact about the
instruments. The full list of claims we are not entitled to make on a positive result is in
[PREREG_gap_map.md](PREREG_gap_map.md) section 0, written before any run.

## Provenance

The mechanistic chain is ported from
[recipient-probe](https://arxiv.org/abs/2607.03598) (*They Infer What You Meant: Models Represent
Communicative Intent More Reliably Than They Act On It*), which established represents, discards,
recovers on six models across four families. Reused from that work:

- the leave-phrasing-out probe with a bag-of-words baseline and permutation test,
- the steering rig and its dose-response protocol,
- the norm-matched random-direction control battery,
- the crossed 2x2 design from `modal_valence.py`, which already showed a valence axis decodable at
  0.90 and 0.92 on Qwen2.5-3B and Llama-3.1-8B, near-orthogonal to the intent axis (cosine 0.083 and
  0.156).

That prior valence axis is a property of the *stimulus*, not a state of the model, which is exactly
the confound this design removes by holding the prompt constant.

Paper discipline, prereg format, and the runnable checks come from `paper-harness`.

## Layout

```
PREREG_gap_map.md      the first design, retained unedited. Its own controls killed it.
PREREG_readout_gap.md  the current design, frozen. What the confirmatory run answers to.
src/report_gap/        stimuli, direction fitting, injection hooks, judge-free scorers,
                       the planted-discrepancy control, and the analysis primitives
experiments/           the Modal runs, the confirmatory runner, and the CPU selftest
results/               recorded outputs, including the nulls
data/sweeps/           raw sweep json
tests/                 pipeline guards and the negative tests for every scorer
paper/                 the report
```

## Running the confirmatory arm

```bash
# 1. select each model's alpha band. section 6 freezes a RULE, not a number, because a single
#    alpha is a different intervention on different models: at 0.025 a positive injection moves
#    positive-option mass +0.056 on Qwen2.5-1.5B and +0.43 on Qwen2.5-3B.
modal run experiments/modal_alpha_recal.py                  # the two non-eval calibrators
modal run experiments/modal_alpha_recal.py --which eval     # writes data/sweeps/band_*.json

# 2. the confirmatory arm. refuses to start without a band file, and refuses a model the band
#    file marks inert.
modal run experiments/modal_readout.py --smoke              # 3 items, one wording, wiring check
modal run experiments/modal_readout.py --model qwen3b       # full frozen matrix

# 3. score it
modal volume get report-gap-data qwen3b ./data/
python experiments/analyze_readout.py data/qwen3b/readout.jsonl
```

`--model llama8b` is expected to refuse. The band file records that model as inert and the runner
stops with the numbers in the message, rather than spending the budget measuring a discrepancy in
a quantity that does not move.

The run streams to a Modal volume and commits after every batch, so an interrupt costs one batch
and a rerun resumes from what is already there. `modal_readout.py` computes nothing;
`analyze_readout.py` holds every endpoint and both instrument gates. That split is the prereg's
rather than a convenience: a runner that also decides is a runner that can decide differently once
it has seen the numbers.

Two things `analyze_readout.py` refuses to do. It will not score a torn artifact, because an
interrupted run is resumable and scoring the fragment is a choice about which cells to keep. And it
will not read the held-out probe wording until the two-wording result is on disk with a timestamp.

## Status

`PREREG_readout_gap.md` frozen 2026-07-31, five deviations logged, `PREREG CLEAN` against the
`paper-harness` checker. 202 tests pass.

The design has narrowed from two models and two poles to **one model, one pole**, and every step of
that narrowing came from a headroom or responsiveness measurement taken before a confirmatory cell
was run:

- **Qwen2.5-3B, positive pole** is the one arm with an instrument. Band 0.002 to 0.020, grid
  (0, 0.002, 0.005, 0.0075, 0.010), own-pole mass rising +0.011, +0.027, +0.057, +0.069.
- **Qwen2.5-3B, negative pole** is inert on its own pole, so the co-primary is `uninformative`.
- **Llama-3.1-8B** is inert entirely and gets no confirmatory arm.

The first two deviations came from a line-by-line audit against
`paper-harness/checklists/CONTROLS.md` that found eight gaps behind a cross-check claim which had
not actually been performed. The remaining three came from the instrument refusing to behave, which
is the point of having one.

See [PLAN.md](PLAN.md) for what is built before the window opens and what is deliberately left
for it.

## License

Apache 2.0.
