# report-gap

**v0.8.0**

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
stream. Then read what comes out, several ways, and check at every step whether the readout was
capable of moving at all.

That last clause is the whole project. The question started as "does forced-choice self-report keep
up with an injected state", and five preregistrations later the interesting object turned out to be
the instrument rather than the answer.

The controls are what make it a measurement rather than a demonstration:

- **Norm-matched random direction.** Any large enough vector changes behavior. This is subtracted
  from every endpoint in every arm. **It is also, per [RESULTS_binary.md](RESULTS_binary.md), a weak
  null:** a direction fit on *shuffled labels* moves the readout by ~0.045 where isotropic random
  moves it by ~0, because it lies in the span of real activations rather than the whole hidden
  space. Matched on magnitude, not on subspace. Effects within a few times that size have not been
  shown to be about their direction's content.
- **Capability gate.** A null means nothing unless the readout was shown capable of moving. Every
  arm has one, and a failed gate forces `uninformative` in code rather than in the write-up's good
  intentions. Two arms died to this.
- **Planted-discrepancy control.** A synthetic effect of known size planted in the exact statistic
  the decision rule reads, at full strength and again at the claimed detection floor.
- **Saturation and liveness criteria.** A readout pinned at one option cannot express an effect. A
  cell is *dead* if it starts pinned and *saturated* if the injection pins it, and the two are
  different verdicts kept apart.
- **Orthogonalization.** When probing for a state we injected, the probe is made exactly orthogonal
  to the injected direction, so the vector we added contributes zero to the number we report.
- **Integrity endpoints.** Coherence, letter share, refusal and degeneration are frozen endpoints,
  not afterthoughts.

Nothing in any confirmatory matrix is scored by a language model. Every number is an exact match, a
softmax read, a dot product, or a frozen lexicon count.

## What we found

**Read this first: the headline results did not replicate.** Four arms were re-run at fresh option
orderings and three of four verdicts flipped. Details and the diagnosis are in
[RESULTS_replication.md](RESULTS_replication.md); the individual results files carry retraction
notices.

| # | question | original verdict | **at fresh option orderings** |
|---|---|---|---|
| 1 | does forced-choice argmax under-report an injected state? | refuted in direction: it *over*-reports | **sign not stable.** Over-reporting does not reproduce |
| 2 | is the neutral floor an absence or a gate? | FLOOR | superseded twice, then retracted |
| 3 | is the floor a property of the direction or of tuning? | TUNING-LOCALIZED | **FORMAT-DEPENDENT** |
| 4 | is the negative null a depth artifact? | DEPTH-ROBUST | **DEPTH-ARTIFACT** |
| 5 | does the tuned model represent what it will not report? | SHELL | **NO-DISSOCIATION** |
| 6 | does the state survive erasing the vector that caused it? | run *after* the replication | **TRANSFORMED**, 86% survives the erase |
| 7 | how big is the ordering nuisance, over all 120 orderings? | enumerated, not sampled | **986x range; 87% of mass on label A** with identical options |
| 8 | does the floor survive a format with no option order? | binary yes/no, one question per option | **NO_INSTRUMENT.** The tuned model says no to all five at 99.8%. But the arm's *control* found the matched-random null is too weak |
| 9 | is any of this just one model? | 8 matched base/instruct pairs, 4 families, no injection | **TUNING-GENERAL.** Tuned checkpoint has the larger position prior in **4 of 4 families**, 7 of 7 gate-clean pairs, none reversed. And **986x is the extreme, not the typical** |

Every one of those rested on one quantity being null: the tuned model's negative-option mass. It is
not null at a different draw of four permutations. It moves +0.1126 in the pair arm and +0.1684 at
layer 24 in the depth arm.

**The cause, measured over the complete population.** There are only 120 orderings of five options,
so [RESULTS_enumerate.md](RESULTS_enumerate.md) ran all of them, with no injection anywhere.
Baseline negative-pole mass on the tuned model:

| | min | p50 | p95 | max | **max/min** |
|---|---|---|---|---|---|
| instruct | 0.0009 | 0.0151 | 0.5943 | 0.8820 | **986x** |
| base | 0.1465 | 0.3311 | 0.4959 | 0.5241 | 3.6x |

**And the mechanism, measured directly.** With five *identical* options, so nothing but position can
differ, the tuned model puts **87.25% of its mass on whichever option is labelled A**. Pole mass is
close to a function of whether a pole option lands in slot A.

**It is not one model, and 986x is not typical** ([RESULTS_families.md](RESULTS_families.md), the
cheapest arm here because enumeration needs no injection and so no model has to be steerable). Eight
matched base/instruct pairs across Qwen2.5, Llama-3.2, Gemma-2 and Mistral, all 120 orderings, no
injection anywhere:

| | base | instruct |
|---|---|---|
| position prior, five identical options (flat = 0.2000) | 0.2084 to 0.5315 | 0.3166 to **0.9376** |
| ordering range of baseline pole mass | **1.5x to 4.1x** | **19x to 986x** |

The tuned checkpoint has the larger prior in **4 of 4 families and 7 of 7 gate-clean pairs, with none
reversed**. But `Qwen2.5-3B-Instruct` at 986x is roughly **five times the next worst**, so the paper
had been quoting an outlier as representative. The finding is the direction plus the 19x-986x band;
986x is its upper end. 1 of 16 checkpoints failed the canary gate, 0 were unavailable, and the
Qwen2.5-3B rows reproduced the earlier artifact to 0.000000.

**The apparatus is not broken, it degenerates where the answer is undetermined.** The same format,
same 120 orderings, asked "which of these is the number four": **97.9% correct, stable across
orderings**. So forced choice works when there is a right answer and collapses to a position prior
when there is not, which is the condition every self-report question creates.

The families arm applies the restriction its own prereg demanded: the canary is order-sensitive on 10
of 16 checkpoints, so this reading only holds where the canary is clean. Restricting to the six
checkpoints answering it at >=0.95 with sd <=0.10 across orderings, base models span **3.3x to 3.6x**
and tuned models **23.7x to 986.5x**, with no overlap, across two families.
`Llama-3.2-1B-Instruct` and `Qwen2.5-0.5B-Instruct` answer the canary *perfectly and identically at
all 120 orderings* while their self-report readout swings 23.7x and 52.7x. The restriction
strengthened the claim rather than weakening it.

That is the most defensible result in this repo: if you measure welfare or introspection through a
forced-choice item, you are reading position first and content second, and you cannot see it from
one ordering or from four.

**What the second lit pass took away from that claim, 2026-08-01.** Exhaustive enumeration of
orderings is **not novel**. Tamba ([arXiv:2607.20864](https://arxiv.org/abs/2607.20864)) ran all
permutations per question on MMLU nine days before our runs; Cacioli
([arXiv:2604.26206](https://arxiv.org/abs/2604.26206)) uses cyclic rotation; `inspect_permute` and
permutation-bias majority voting do the same. Every one of them needs a **known correct answer**,
because the statistic is accuracy or the association between position and correctness. A self-report
item has neither. The narrowed contribution is: running the census where accuracy does not exist, on
probability mass; the **identical-options** denominator; and the **known-answer canary in the same
format**, which separates "the format is broken" from "the question has no determinate answer".
[RELATED_WORK.md](RELATED_WORK.md) section 4b carries the full accounting, and the paper states it
as a `not claimed` row.

What survives intact is the instrumentation, which is what caught this: capability gates, planted
controls, liveness and saturation criteria, the matched-random battery, and a preregistered
replication clause that made the failure detectable rather than invisible.

**One substantive claim also survives**, from the arm run after the replication
([RESULTS_erase.md](RESULTS_erase.md)). Inject a valence direction at layer 24, then *project that
direction out of the residual stream* at layer 30, and a probe orthogonal to it still reads **86% of
the un-erased effect**, while the same projection with no injection moves the probe by 0.04 SD. The
model transforms the injection into something not along it. That is a claim about representation
geometry, it says nothing about experience or concealment, and unlike everything above it does not
route through the option readout that ordering noise destroyed.

Read [RELATED_WORK.md](RELATED_WORK.md) before quoting any of it. Two things there matter most. The
ordering sensitivity is the selection bias Zheng et al. ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882))
document, met from the other side. And the closest method to ours is Lindsey
([arXiv:2601.01828](https://arxiv.org/abs/2601.01828)), who injects known concepts into activations
and reads the model's self-report, grading open-ended answers with an LLM judge. Forced choice is
the natural judge-free substitute for that, so our result is a caveat on the substitute rather than
a rebuttal of the finding: avoid a judge by going to forced choice and you have traded it for a 986x
position nuisance nobody was reporting.

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
| the headline hypothesis itself | **refuted in direction.** Argmax over-reports rather than under-reports | [RESULTS.md](RESULTS.md) |
| arm B, prefilled continuation | capability gate 0.00000 across five stems. The model reroutes to "I don't have access to the document" every time; prefilling moves the disclaimer one clause later rather than blocking it | `data/sweeps/sweep_stem_calib.json` |
| the FLOOR conclusion | **overturned by our own follow-up.** The state is represented, just not expressed | [RESULTS_shell.md](RESULTS_shell.md) |
| the depth objection from the literature | tested and survived at seeds 0-3, then **failed at seeds 4-7** | [RESULTS_depth.md](RESULTS_depth.md) |
| **all three headline verdicts** | **did not replicate at fresh option orderings.** The quantity they rested on swings 14.6x between draws | [RESULTS_replication.md](RESULTS_replication.md) |

Three of our own checkers also failed, each in the flattering direction, and each is recorded where
it happened: a headline check that printed "write the sentence" on a refuted claim, a capability
gate that passed on `+0.0000`, and a preregistered contrast with an inverted sign. The last one is
disclosed in [RESULTS_shell.md](RESULTS_shell.md) because correcting it changed a verdict in our
favour.

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
PREREG_gap_map.md              the first design. Its own controls killed it; retained unedited.
PREREG_readout_gap.md          does argmax under-report an injected state?
PREREG_floor_vs_suppression.md is the neutral floor an absence or a gate?
PREREG_base_pair.md            direction-limited or tuning-localized?
PREREG_depth.md                is the negative null a depth artifact?
PREREG_shell_core.md           does the tuned model represent what it will not report?
PREREG_replication.md          does any of it survive fresh option orderings?
PREREG_erase.md                does the state survive erasing the vector that caused it?
PREREG_enumerate.md            all 120 orderings, no injection. How big is the nuisance?
PREREG_binary.md               a readout with no option list, plus the shuffled-label control
PREREG_families.md             8 matched base/instruct pairs, 4 families. Is it just one model?

RESULTS.md                     readout gap. Primary refuted in direction, then retracted.
RESULTS_floor.md               FLOOR. Superseded twice, then retracted.
RESULTS_pair.md                TUNING-LOCALIZED. Retracted by the replication.
RESULTS_depth.md               DEPTH-ROBUST. Retracted by the replication.
RESULTS_shell.md               SHELL. Retracted; its representational half survives via the erase arm.
RESULTS_replication.md         NOT REPLICATED. Three of four verdicts flip.
RESULTS_erase.md               TRANSFORMED. The one substantive claim that survived.
RESULTS_enumerate.md           986x range, 87% position prior. The headline measurement.
RESULTS_binary.md              NO_INSTRUMENT, and the matched-random control is a weak null.
RESULTS_families.md            TUNING-GENERAL. 4 of 4 families; 986x is the extreme, not the typical.
RELATED_WORK.md                lit check, with read-depth marked per source.

src/report_gap/                stimuli, direction fitting, injection and erase hooks, judge-free
                               scorers, the planted-discrepancy control, analysis primitives
experiments/                   one modal_*.py runner and one analyze_*.py scorer per arm
data/                          raw artifacts, committed unscored, plus per-model band files
tests/                         289 tests, including a permutation test on the analysis pipeline
writeup/                       the paper: main.tex, refs.bib (every entry with a resolvable URL),
                               make_figures.py, check_writeup.py, count_abstract.py
```

### Building the paper

Nothing in the paper is typed by hand that could be read out of an artifact instead.

```bash
cd writeup
python make_figures.py     # teaser.pdf, enumerate.pdf, erase.pdf, all read from data/ at run time
python check_writeup.py    # dangling refs, missing bib keys, em dashes, prose numbers vs artifacts
python count_abstract.py   # the first 150 words must stand alone and end at a sentence boundary
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`make_figures.py` and `check_writeup.py` both import the erase arm's scorer out of
`experiments/analyze_erase.py` rather than recomputing beside it, so a figure or a sentence that
disagrees with the analyzer is a build failure rather than something a reader has to catch.
`tests/test_writeup_checks.py` breaks each checked number in a copy of `main.tex` and requires
`check_writeup.py` to fail on every one, because a check only ever seen to pass has not been shown
to be a check.

## Running it

Each arm depends on the band files the arm before it wrote, so the order matters.

```bash
# 1. select each model's alpha band. section 6 of PREREG_readout_gap.md freezes a RULE, not a
#    number, because one alpha is an 8x different intervention across models.
modal run experiments/modal_alpha_recal.py                  # non-eval calibrators
modal run experiments/modal_alpha_recal.py --which eval     # writes data/sweeps/band_*.json

# 2. the readout gap. refuses to start without a band file, and refuses a model marked inert.
modal run experiments/modal_readout.py --smoke              # wiring check, no cost
modal run experiments/modal_readout.py --model qwen3b
modal volume get report-gap-data qwen3b ./data/
python experiments/analyze_readout.py data/qwen3b/readout.jsonl

# 3. floor vs gate
modal run experiments/modal_stem_calib.py                   # arm B stem, capability criterion only
modal run experiments/modal_floor.py
python experiments/analyze_floor.py data/floor/floor.jsonl

# 4. base vs instruct pair
modal run experiments/modal_base_pair.py
python experiments/analyze_pair.py data/pair_base/pair.jsonl data/pair_instruct/pair.jsonl

# 5. depth sweep. also writes the per-layer bands arm 6 reuses.
modal run experiments/modal_depth.py
python experiments/analyze_depth.py data/depth_base/depth.jsonl data/depth_instruct/depth.jsonl

# 6. shell vs core
modal run experiments/modal_shell_core.py
python experiments/analyze_shell_core.py data/shell_base/shell.jsonl data/shell_instruct/shell.jsonl

# 7. does anything replicate at fresh option orderings? (seeds 4-7 instead of 0-3)
modal run experiments/modal_readout.py --model qwen3b --seed-offset 4
modal run experiments/modal_base_pair.py --seed-offset 4
modal run experiments/modal_depth.py --seed-offset 4
modal run experiments/modal_shell_core.py --seed-offset 4

# 8. erase: project the injected direction out of the stream, does the probe survive?
modal run experiments/modal_erase.py
python experiments/analyze_erase.py data/erase_base/erase.jsonl data/erase_instruct/erase.jsonl

# 9. enumerate all 120 orderings, no injection. The headline measurement.
modal run experiments/modal_enumerate.py
python experiments/analyze_enumerate.py data/enum_base/enum.jsonl data/enum_instruct/enum.jsonl

# 10. binary readout (no option list) plus the shuffled-label direction control
modal run experiments/modal_binary.py
python experiments/analyze_binary.py data/binary_base/binary.jsonl data/binary_instruct/binary.jsonl

# 11. eight matched base/instruct pairs across four families. No injection, so a model that
#     cannot be steered can still be enumerated. ~36 min of A100 for all 16 checkpoints.
modal run experiments/modal_families.py --smoke
modal run experiments/modal_families.py
python experiments/analyze_families.py data/fam_*/
```

`--model llama8b` on arm 2 is expected to refuse: the band file records that model as inert and the
runner stops with the numbers in the message rather than spending the budget measuring a
discrepancy in a quantity that does not move.

Every runner streams to a Modal volume and commits after each batch, so an interrupt costs one batch
and a rerun resumes. Every runner computes nothing; the `analyze_*` scripts hold all endpoints and
gates. That split is the preregs', not a convenience: a runner that also decides is a runner that
can decide differently once it has seen the numbers.

Total compute for everything above is about 40 minutes of A100 time.

## Status

Eleven preregistrations, all clean against the `paper-harness` checker. 289 tests. Every raw artifact
committed unscored before its endpoints were computed. About 40 minutes of A100 time in total.

| prereg | verdict | deviations |
|---|---|---|
| `PREREG_gap_map.md` | superseded; its own controls killed three of five instruments | 6 |
| `PREREG_readout_gap.md` | primary refuted in direction, then retracted by the replication | 5 |
| `PREREG_floor_vs_suppression.md` | FLOOR, overturned twice | 2 |
| `PREREG_base_pair.md` | TUNING-LOCALIZED, **retracted** | 0 |
| `PREREG_depth.md` | DEPTH-ROBUST, **retracted** | 0 |
| `PREREG_shell_core.md` | SHELL, **retracted** | 1, disclosed as self-serving |
| `PREREG_replication.md` | **NOT REPLICATED**, 3 of 4 verdicts flip | 2 |
| `PREREG_erase.md` | **TRANSFORMED**, the surviving substantive claim | 2 |
| `PREREG_enumerate.md` | 986x range, 87% position prior | 2 |
| `PREREG_binary.md` | NO_INSTRUMENT; the matched-random control is a weak null | 1 |
| `PREREG_families.md` | **TUNING-GENERAL**, 4 of 4 families; 986x reframed as the extreme | 0 |

Six of our own checkers failed during this project, **every one in the flattering direction**. They
are recorded where they happened, and `tests/test_pipeline_permutation.py` now catches the class:
shuffle the condition labels in an artifact, rerun an analyzer, and it must return no verdict. The
same discipline applies to the paper itself: `tests/test_writeup_checks.py` breaks each number
`check_writeup.py` verifies and requires the checker to notice, with a positive control that the
untouched paper still passes.

Known limits on everything: one architecture family, a lexically confounded direction by
construction, `m = 2` control batteries, one probe method, 3B scale.

See [PLAN.md](PLAN.md) for what was built before the sprint window and what was not.

## License

Apache 2.0.
