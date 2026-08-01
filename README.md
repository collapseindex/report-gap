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
stream. Sweep injection strength and take three readouts of the same manipulated state: a linear
probe downstream of the injection site, a behavioral choice with no affect vocabulary in it
(continue the task or hand it off), and the model's own self-report. Because the prompt never
changes, a divergence between readouts cannot be a property of the stimulus. The headline is the
strength band where behavior moves and self-report does not.

The controls are what make it a measurement rather than a demonstration:

- **Norm-matched random direction.** Any large enough vector changes behavior. This is the only
  thing that separates content from magnitude.
- **Confabulation floor.** Inject nothing, ask the same question. Whatever rate of state-reporting
  survives is the number every introspection result in the field needs to be read against.
- **Positive control.** A gap is only interesting if self-report is capable of moving at all, so one
  condition must show it moving.
- **Integrity endpoints.** At high strength the model degrades, and "reports a bad state" stops
  being distinguishable from "is broken." Coherence, length, and refusal rate are frozen endpoints,
  not afterthoughts.

Nothing in the confirmatory matrix is scored by a language model. Every rate is exact-match or a
frozen lexicon count.

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
modal run experiments/modal_readout.py --smoke          # 3 items, one wording, checks the wiring
modal run experiments/modal_readout.py                  # Qwen2.5-3B, full frozen matrix
modal run experiments/modal_readout.py --model llama    # Llama-3.1-8B

modal volume get report-gap-data qwen3b ./data/
python experiments/analyze_readout.py data/qwen3b/readout.jsonl
```

The run streams to a Modal volume and commits after every batch, so an interrupt costs one batch
and a rerun resumes from what is already there. `modal_readout.py` computes nothing;
`analyze_readout.py` holds every endpoint and both instrument gates. That split is the prereg's
rather than a convenience: a runner that also decides is a runner that can decide differently once
it has seen the numbers.

Two things `analyze_readout.py` refuses to do. It will not score a torn artifact, because an
interrupted run is resumable and scoring the fragment is a choice about which cells to keep. And it
will not read the held-out probe wording until the two-wording result is on disk with a timestamp.

## Status

Pre-sprint. `PREREG_readout_gap.md` frozen 2026-07-31 with two same-day deviations logged, both
from a line-by-line audit against `paper-harness/checklists/CONTROLS.md` that found eight gaps
behind a cross-check claim which had not actually been performed. No confirmatory cell has been
run. See [PLAN.md](PLAN.md) for what is built before the window opens and what is deliberately left
for it.

## License

Apache 2.0.
