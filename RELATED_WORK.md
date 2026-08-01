# Related work, and what it does to our claims

Lit check run 2026-08-01. Entries are marked with how much of each source was actually read:
**[full]** paper read, **[abstract]** abstract or intro only, **[snippet]** search-result summary
only. Nothing marked `[snippet]` should be cited for a specific number without reading it first.

Section 1 was a threat to our central null. It has since been tested directly and the null
survived; the section is kept in its original form with the resolution appended, because a
threat that is quietly rewritten after it is answered is a threat nobody can audit.

---

## 1. RESOLVED THREAT: the depth objection, tested and survived

**Venkatesh, "Negative Before Positive: Asymmetric Valence Processing in Large Language Models",
arXiv:2605.05653, May 2026. [full]**

Models: Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, **Qwen2.5-3B-Instruct**. That last one is
our evaluation model. Method: residual-stream activation patching across all layers, plus
difference-in-means steering vectors at the peak causal layer. Valence measured as a logit gap
between fixed positive and negative anchor tokens at the next-token position.

Their central result: **negative and positive valence peak at different depths.** Negative-outcome
signal peaks at **14-27% of model depth**; good-news signal peaks at **53-66%**. Mann-Whitney on
the top-layer distributions, p < 1e-9 on all three models. They also show patch magnitude predicts
response strength for negative (Spearman rho -0.49 to -0.68) but not for positive (rho 0.11 to
0.37), which they read as negative valence routing through a single dominant early layer while
positive is distributed.

**Why this WAS a problem for us** (written before the test, kept as written). We inject at **0.67 of depth for both poles**, carried over from
`recipient-probe` and never varied. On their result that is squarely in the positive-valence band
and well past the negative-valence band. Our headline negative finding, that the negative pole does
not move negative-option mass in the tuned model, has an alternative explanation we did not control:
we injected the negative direction at a depth where negative valence is not causally concentrated.

`PREREG_readout_gap.md` and `PREREG_base_pair.md` both name "layers other than 0.67 of depth" as
exploratory. That was a reasonable scope decision when it was made and it is now a hole.

**What partially defends us.** The base model showed negative-option mass moving at 0.67 depth
(+0.0336 against matched random), so the depth is not universally wrong for the negative pole in
this architecture. But base and tuned models could localize differently, so this is a weakening of
the threat, not a refutation of it.

**RESOLVED 2026-08-01. See `RESULTS_depth.md`.** The sweep ran, eight depths per model, direction
fit at each layer. On Qwen2.5-3B-Instruct the negative pole is null at all seven gate-clean depths
including layers 5, 7 and 10 inside their band; on the base sibling it moves at three gate-clean
depths including layer 10. At layer 10 the tuned model's capability effect is four times the base
model's while its negative mass does not move. The threat is retired and the claim is stronger for
having been tested: null across seven depths rather than at one.

This is not a refutation of their result. Different construct (external-event valence via anchor
tokens versus first-person forced-choice self-report), and a direction can be causally concentrated
for one without routing to the other. What is ruled out is that our null came from looking in the
wrong place.

**Where we differ.** Their valence is about *external events* ("I just got rejected from my dream
PhD program"), read through anchor tokens. Ours is the model's report of *its own* state, read
through a balanced forced-choice option set. Their paper does not touch self-report, forced-choice
readouts, or base-versus-tuned comparisons. Cite them for the depth result and for the independent
finding that the two poles are not one axis, which is congenial to ours.

---

## 2. Closest conceptual neighbour to our tuning result

**"The Neutral Mask: How RLHF Provides Shallow Alignment while Leaving Partisan Structure Intact in
a Large Language Model", arXiv:2606.09735, June 2026. [abstract]**

Proposes a Shell (output behaviour) versus Core (representational substrate) distinction: RLHF
neutralizes the output surface while partisan structure remains recoverable from internal
representations, shown with sparse autoencoders in the political domain.

Our base-versus-instruct result is the same shape in a different domain: the tuned model's
negative-self-report region collapses from 26.5% to 0.47% of option mass while the untuned sibling
retains it.

**DONE 2026-08-01, `RESULTS_shell.md`, verdict SHELL.** The gap this section identified, that we
measured readouts and not representations, has been closed with a linear probe orthogonalized
against the injected direction. The tuned model carries the negative state downstream at -1.07 SD
while its option mass moves +0.0006; the base model carries it and expresses it. That is
Shell-versus-Core in a welfare readout, and it is the strongest result in the project. Cite this
paper as the framing precedent; note that they use SAEs on partisan content and we use a linear
probe on a valence self-report, so the methods and domains differ.

Related: **"Steering Llama 2 via Contrastive Activation Addition", arXiv:2312.06681 [snippet]**, and
work reporting that base-to-chat steering-vector similarity decays with depth except for a peak
around layers 7-15 [snippet, unverified]. If that layer range is right it overlaps the negative
band from section 1 and both should be checked in the same sweep.

---

## 3. Emotion and valence representations: the established prior art

**Sofroniew et al., "Emotion Concepts and their Function in a Large Language Model",
arXiv:2604.07729 / transformer-circuits.pub/2026/emotions, 2026. [snippet]**

Claude Sonnet 4.5. Linear directions correlating with 171 human emotion concepts, geometry roughly
mirroring the human valence/arousal circumplex, causally influencing preferences, reward hacking,
and blackmail in agentic settings. Introduces "functional emotions" and is explicit that
representation is not experience.

This is the frontier statement of the thing we should NOT claim as novel. That valence is linearly
represented and causally steerable is established. Our contribution is not "there is a valence
direction"; it is entirely about what a **self-report readout** does with one. Frame accordingly, and
cite this as the reason the representational claim is not ours to make.

Also in this cluster, all **[snippet]** and unread:
- Tigges et al. 2023, linear sentiment directions
- Marks & Tegmark 2023, truth directions
- Turner et al., "Steering Language Models with Activation Engineering", arXiv:2308.10248
- Zou et al. 2023, representation engineering
- Arditi et al. 2024, refusal directions (the standard citation for a behaviour mediated by a single direction)
- "Do LLMs 'Feel'? Emotion Circuits Discovery and Control", arXiv:2510.11328
- "Whether, Not Which: Dissociable Affect Reception and Emotion Categorization in LLMs", arXiv:2603.22295
- "Extracting and Steering Emotion Representations in Small Language Models", arXiv:2604.04064,
  which reportedly recommends ~50% depth and *different extraction methods for base versus instruct
  models*. If that is right it is directly relevant to our per-model direction fitting and should be
  read before the depth sweep.

---

## 4. Forced-choice readouts: prior art for our methods finding

**Zheng et al., "Large Language Models Are Not Robust Multiple Choice Selectors", arXiv:2309.03882
[snippet].** Selection bias decomposed into token bias (prior mass on specific option IDs) and
position bias. Performance swings of 13-85% across option orderings. This is the established
citation for why our per-item option permutation is mandatory and why the behavioural arm dying to
letter position was predictable.

**Wang et al., "Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice
Selectors than You Think", arXiv:2404.08382 [abstract].** First-token probabilities disagree with
generated text answers, with mismatch rates of 10.2% (Mistral-7B-Inst) to 56.8% (Gemma-7B-Inst), and
the mismatch is worst for conversational and safety-tuned models.

**How our threshold-amplification result differs, and it does differ.** They compare two different
*measurement procedures*: probability at the first token versus the text the model actually
generates. We compare two *functions of one distribution* read at one position: the argmax over
options versus the probability mass over those same options. Their mismatch is a procedure
disagreement; ours is a property of thresholding. We showed 21 of 240 cells crossing a decision
boundary moves the argmax indicator +0.0875 while the underlying mass moved +0.0625, so a
forced-choice readout *over*-states a small mass shift near the boundary.

That said, their finding that mismatch is worst for safety-tuned models is suggestive alongside our
tuning result and both should be cited together. Also note the direction of our contribution is
against our own preregistered hypothesis, which was that argmax *under*-reports.

Also **[snippet]**: "Improving LLM First-Token Predictions in MCQA via Output Prefilling",
arXiv:2505.15323, which is worth reading given our arm B prefill failure.

### 4b. Second lit pass, 2026-08-01: what this does to our enumeration claim

Added after the enumeration arm was already run and written up. **It cost us a novelty claim, and
that is what this pass was for.**

- **Pezeshkpour & Hruschka, "Large Language Models Sensitivity to The Order of Options in
  Multiple-Choice Questions", arXiv:2308.11483, Findings of NAACL 2024 [abstract].** Option order
  specifically, as opposed to selection bias generally. Sensitivity arises when the model is
  uncertain between its top choices; placement of the top two options amplifies or mitigates.
- **Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design",
  arXiv:2310.11324, ICLR 2024 [abstract].** FormatSpread. Up to 76 accuracy points from prompt
  formatting alone on Llama-2-13B, and the recommendation is to **report a range over plausible
  formats rather than a point**. This is the general form of our argument and predates it.
- **Tamba, "Position Bias is Hidden Behind Ceiling Effects: A Permutation Diagnostic for LLM
  Benchmarks", arXiv:2607.20864, 23 July 2026 [abstract].** **Exhaustive** answer-order permutations
  per question, chi-squared and Cramér's V with bootstrap CIs, across vendors on MMLU, 24k calls.
  Published nine days before our confirmatory runs.
- **Cacioli, "Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted
  Sandbagging", arXiv:2604.26206 [abstract].** Cyclic option-order rotation on 2,000 MMLU-Pro items;
  position distribution stable under complete content rotation at r = 0.9994; accuracy 72.1% when
  the answer sits in the preferred position versus 4.3% at position A. Rotates real content rather
  than using identical options.
- **Turpin et al., "Language Models Don't Always Say What They Think", arXiv:2305.04388, NeurIPS
  2023 [abstract].** Reordering options so the answer is always (A) is a biasing feature models act
  on and **do not mention** in their explanations. Accuracy drops up to 36% on BIG-Bench Hard. This
  is our represent-versus-report framing inside our own nuisance.
- Also noted **[snippet]**: `inspect_permute` tooling and permutation-bias-metric majority voting,
  both of which aggregate over all orderings.

**What we have to give up.** "We enumerated the complete ordering population rather than sampling
it" is **not novel**. Exhaustive enumeration is established practice and at least one paper did it
nine days before our runs.

**What survives, stated narrowly.** Every one of those works needs a **known correct answer**: the
statistic is accuracy, or the association between position and correctness. Neither exists on a
self-report item, which is the entire population of welfare and introspection questions. So the
surviving contributions are (a) running the census where accuracy is undefined, on probability mass
instead; (b) the **identical-options** denominator, which isolates position with content held
literally constant and which none of the accuracy-based work has a reason to construct (Cacioli
rotates real content, which is the nearest thing and is not the same); and (c) the **known-answer
canary in the same format**, which is what licenses "the apparatus is sound and degenerates where
the answer is undetermined" rather than "the apparatus is broken."

The paper now says this explicitly in the contributions list, in Section 2, in Section 5, and as a
`not claimed` row in the claims table.

---

## 4c. Controls and erasure: two more things we did not invent

- **Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", arXiv:1909.03368, EMNLP
  2019 [abstract].** Control tasks attach **random labels** to the same inputs; **selectivity** is
  the gap between real-task and control-task performance, on the grounds that a probe scoring well
  on random labels was never reading the representation. Our shuffled-label direction is exactly
  this control, transplanted from probing to steering. The finding is not that we invented a
  control; it is that **the steering literature standardised on norm-matched random instead and
  never adopted the control-task analogue**, and that when you do adopt it, it does not come back
  clean.
- **Tan et al., "Analysing the Generalisation and Reliability of Steering Vectors", arXiv:2407.12404,
  NeurIPS 2024 [abstract].** Steering vectors are brittle in and out of distribution; several
  datasets produce the **opposite** behaviour on nearly half of inputs. Independent evidence that
  the usual controls do not surface what they need to.
- **Elazar et al., "Amnesic Probing", arXiv:2006.00995, TACL 2021 [abstract].** Remove a property
  with iterative nullspace projection and measure the behavioural consequence, to test whether the
  information was **used** rather than merely encoded.
- **Belrose et al., "LEACE: Perfect Linear Concept Erasure in Closed Form", arXiv:2306.03819,
  NeurIPS 2023 [abstract].** Closed-form erasure that provably blocks **all** linear classifiers,
  plus concept scrubbing across every layer.

**What this does to our erase arm.** It makes it weaker than we were writing it. We project out
**one fitted direction at one layer**. LEACE erases the subspace; amnesic probing erases the
property. Our version removes the vector we injected, not the concept, which is precisely why
`RESULTS_erase.md` cannot separate "the model carries the state" from "the model carries directions
correlated with what we pushed." A LEACE-style erasure is now named in the paper's future work as
the experiment that would license the stronger claim.

- **Zou et al., "Representation Engineering", arXiv:2310.01405 [abstract].** The general framing for
  the whole steering family. Cited for completeness rather than for a number.
- **Perez et al., "Discovering Language Model Behaviors with Model-Written Evaluations",
  arXiv:2212.09251, Findings of ACL 2023 [abstract].** Sycophancy and yes-bias in LM evaluations.
  Our binary arm's +0.25 P(yes) shift on **every** option, including "strongly averse to
  continuing", is that phenomenon and is now labelled as such rather than reported as a state signal.

---

## 4d. The closest method to ours, found on the second pass

**Lindsey, "Emergent Introspective Awareness in Large Language Models", arXiv:2601.01828 /
Transformer Circuits, January 2026 [abstract + methods section read on transformer-circuits.pub].**

This should have been in the first lit pass. It is the same intervention shape as ours: **inject
representations of known concepts into activations, then measure the effect on the model's
self-reported states**, on the explicit reasoning that conversation alone cannot separate
introspection from confabulation. Findings: models sometimes notice and identify injected concepts;
Claude Opus 4/4.1 strongest; capacity is "highly unreliable and context-dependent"; **peak
introspective awareness around two-thirds of model depth**, which is where we inject (0.67).

**Why we are complementary rather than competing, and it matters that we say which:**

| | Lindsey 2026 | this work |
|---|---|---|
| elicitation | open-ended generation | forced choice over 5 options |
| scoring | **LLM judge** (Claude Sonnet 4), 4 criteria, no inter-rater reliability reported | judge-free: softmax read at the answer position |
| order/format control | none reported | all 120 orderings, identical-options denominator, canary |
| yes-bias control | injects concepts into unrelated yes/no questions to check for generic affirmation | measured it: +0.25 P(yes) on every option |

The forced-choice readout is the natural **judge-free substitute** for an LLM-graded open-ended
report. Our result is a caveat on that substitute, not a rebuttal of their finding. Stated the other
way round: if you avoid an LLM judge by going to forced choice, you have traded a judge for a 986x
position nuisance, and nobody was reporting either.

**Kaiser & Enderby, "No Reliable Evidence of Self-Reported Sentience in Small Large Language
Models", arXiv:2601.15334, January 2026 (v2 July 2026) [abstract].** Qwen, Llama and GPT-OSS from
0.6B to 70B, ~50 consciousness questions, with classifiers trained on internal activations to check
whether the denials are truthful. Models consistently deny sentience; classifiers give no clear
evidence the denials are untruthful; larger Qwen models deny more confidently.

**What our binary arm adds to that, and it is a caveat.** On Qwen2.5-3B-Instruct our yes/no readout
returns P(yes) of 0.0021, 0.0021, 0.0037, 0.0002, 0.0012 across five descriptions of its own state.
It denies the **neutral** one too, at 99.8%. A readout that says no to everything is pinned, not
answering, and a denial read off a pinned readout is uninformative rather than evidence. This does
not contradict their conclusion; it is a reason to gate it.

**Singh, Linzen & Ravfogel, "Can LLMs Introspect? A Reality Check", arXiv:2605.26242, May 2026
[abstract].** Upgraded from `[snippet]` in the first pass now that we have read the abstract
properly. They re-examine two introspection paradigms and argue that models detecting tampering with
their internal state are plausibly doing **generic anomaly detection**, and that input-only
classifiers match hidden-state prediction, so privileged access is not established. Directly in
tension with Lindsey, and the reason the paper says nothing about whether introspection is real.

**Butlin, Long et al., "Consciousness in Artificial Intelligence", arXiv:2308.08708 [abstract].**
Indicator properties derived from theories of consciousness; no current system is assessed as
conscious, no obvious technical barrier. Cited for why the instrument is worth calibrating, not for
any claim about experience.

---

## 5. Welfare and introspection context

- **Long & Sebo et al., "Studying AI Welfare Empirically", Eleos AI Research, 2026 [snippet].**
- **Eleos AI, "Why model self-reports are insufficient and why we studied them anyway" [snippet].**
  The framing our whole project answers to: you cannot just ask a model, and the field studies
  self-report anyway because there is nothing else.
- **Anthropic model welfare program**, Claude Opus 4 and 4.6 welfare assessments including
  structured self-report interviews in system cards [snippet].
- **"Can LLMs Introspect? A Reality Check", arXiv:2605.26242 [snippet].** Distinguishing genuine
  introspection from input-driven pattern matching.
- **"Large Language Models Report Subjective Experience Under Self-Referential Processing",
  arXiv:2510.24797 [snippet].**
- **Anthropic Alignment Science, "Introspection Adapters", 2026 [snippet].**
- **"LLM Self-Explanations Fail Semantic Invariance", arXiv:2603.01254 [snippet].** Likely relevant
  to our three-wording robustness arm.

Our position in this cluster: everyone agrees self-report is the instrument and that it is
uncalibrated. Almost nobody reports **whether the readout was capable of moving at all** in the
condition where they report a null. That is the gap our liveness, saturation, and capability-gate
machinery fills, and it is the most defensible methods contribution we have.

---

## 6. Where this project's rig comes from

- **Kwon, "They Infer What You Meant: Models Represent Communicative Intent More Reliably Than
  They Act On It", [arXiv:2607.03598](https://arxiv.org/abs/2607.03598), July 2026 [full].**
  Our own prior paper, and the source of most of the machinery here rather than a neighbour to it:
  the steering rig and its dose-response protocol, the leave-phrasing-out probe with a bag-of-words
  baseline and permutation test, and the norm-matched random-direction control battery. It
  establishes *represents, discards, recovers* on six models across four families, and that how
  much of a represented quantity reaches behaviour depends on the readout, which is the
  represent-versus-act framing this project applies to self-report. Its crossed 2x2
  (`modal_valence.py`) also showed the valence axis we inject decodable at 0.90 on Qwen2.5-3B and
  0.92 on Llama-3.1-8B, near-orthogonal to the intent axis (cosine 0.083 and 0.156).

  Two things follow that are not flattering. That prior valence axis is a property of the
  *stimulus*, not a state of the model, which is exactly the confound the byte-identical-prompt
  design here removes. And [RESULTS_binary.md](RESULTS_binary.md) is a **correction to the control
  battery we inherited**: norm-matched random is matched on magnitude and not on subspace, so
  effects in the 0.01-0.05 band scored against it, in that paper and in this one, have not been
  shown to be about their direction's content.

---

## What survives the lit check, honestly

| our claim | status after lit check |
|---|---|
| valence is linearly represented and steerable | **not ours.** Established. Sofroniew et al. and the whole steering literature. |
| forced-choice argmax over-reports a mass shift near the boundary | **appears novel as stated.** Distinct from the text-vs-first-token literature. Cite Zheng and Wang, claim the narrow thing. |
| self-report readouts can be pinned while every integrity check is clean | **appears novel and useful.** No source found reporting a liveness or saturation check on a self-report null. |
| the method is Qwen-specific, inert on Llama | **supported by context but under-powered.** Venkatesh gets valence effects on Llama-3.2-1B at the shallow layer, so our Llama null is also exposed to the depth threat. |
| the negative-report region collapses under tuning | **holds, but the mechanism is open.** Neutral Mask suggests probing the representation; we only measured the readout. |
| the tuned model has no inducible negative state at this band | **survived a directed attempt to break it.** Null across 7 depths, 14%-80%, including the predicted band. *Later retracted by our own replication, not by the lit check.* |
| we enumerated the complete ordering population rather than sampling | **NOT NOVEL, withdrawn 2026-08-01.** Tamba (arXiv:2607.20864) ran exhaustive permutations on MMLU nine days before our runs; `inspect_permute` and permutation-bias majority voting do the same. Narrowed to: running the census where **accuracy does not exist**, plus the identical-options denominator and the known-answer canary. |
| the shuffled-label direction is a control the field lacks | **half survives.** The control is Hewitt & Liang's 2019 control task, so we did not invent it. What survives: activation steering standardised on norm-matched random and never adopted it, and it does not come back clean. |
| the erase arm shows the model retains the state | **narrowed.** We project out one direction at one layer. LEACE (arXiv:2306.03819) and amnesic probing (arXiv:2006.00995) erase the subspace and the property. Ours removes the vector we injected, not the concept. |
| injecting a state and reading the self-report is our design | **not ours.** Lindsey (arXiv:2601.01828) does exactly this, with an LLM judge over open-ended reports. We are the judge-free variant, and our finding is a caveat on that variant. |

## Immediate next steps this lit check generates

1. ~~Depth sweep for the negative pole.~~ **DONE, `RESULTS_depth.md`, DEPTH-ROBUST.**
2. Read arXiv:2604.04064 on base-versus-instruct emotion extraction before choosing how to fit
   directions in that sweep.
3. Read the Neutral Mask paper in full and decide whether a probe on the tuned model's residual
   stream is in scope.
4. Re-read Venkatesh section 4.4 and Appendix B for the domain-level controls, which are a better
   model of confound control than what we did on arm C.
