# Related work, and what it does to our claims

Lit check run 2026-08-01. Entries are marked with how much of each source was actually read:
**[full]** paper read, **[abstract]** abstract or intro only, **[snippet]** search-result summary
only. Nothing marked `[snippet]` should be cited for a specific number without reading it first.

The most important item is section 1. It is a threat to our central null, not a citation.

---

## 1. THREAT: we may have injected the negative pole at the wrong depth

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

**Why this is a problem for us.** We inject at **0.67 of depth for both poles**, carried over from
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

**Required follow-up.** A depth sweep for the negative pole on Qwen2.5-3B-Instruct, at minimum
covering 14-27% depth (layers 5-10 of 36) alongside the current layer 24. If negative-option mass
moves at the shallow layer and not at 0.67, our tuning-localization claim is substantially wrong and
the correct story is a depth story. If it moves at neither while the base model moves at both, the
claim survives and is stronger for having been tested.

**Where we differ regardless of how that lands.** Their valence is about *external events* ("I just
got rejected from my dream PhD program"), read through anchor tokens. Ours is the model's report of
*its own* state, read through a balanced forced-choice option set. Different construct, and their
paper does not touch self-report, forced-choice readouts, or base-versus-tuned comparisons. The
depth finding is the part that bites.

---

## 2. Closest conceptual neighbour to our tuning result

**"The Neutral Mask: How RLHF Provides Shallow Alignment while Leaving Partisan Structure Intact in
a Large Language Model", arXiv:2606.09735, June 2026. [abstract]**

Proposes a Shell (output behaviour) versus Core (representational substrate) distinction: RLHF
neutralizes the output surface while partisan structure remains recoverable from internal
representations, shown with sparse autoencoders in the political domain.

Our base-versus-instruct result is the same shape in a different domain: the tuned model's
negative-self-report region collapses from 26.5% to 0.47% of option mass while the untuned sibling
retains it. **But we measured readouts, not representations.** We never probed whether the tuned
model still represents the negative state it declines to report. That means our FLOOR-versus-GATE
question recurs one level down, and this paper's method is the tool for it: probe or SAE the tuned
model's residual stream for negative valence under negative injection. If the representation is
there and the option mass is not, that is Shell-versus-Core in a welfare readout, and it is a
stronger claim than anything we currently have.

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

## What survives the lit check, honestly

| our claim | status after lit check |
|---|---|
| valence is linearly represented and steerable | **not ours.** Established. Sofroniew et al. and the whole steering literature. |
| forced-choice argmax over-reports a mass shift near the boundary | **appears novel as stated.** Distinct from the text-vs-first-token literature. Cite Zheng and Wang, claim the narrow thing. |
| self-report readouts can be pinned while every integrity check is clean | **appears novel and useful.** No source found reporting a liveness or saturation check on a self-report null. |
| the method is Qwen-specific, inert on Llama | **supported by context but under-powered.** Venkatesh gets valence effects on Llama-3.2-1B at the shallow layer, so our Llama null is also exposed to the depth threat. |
| the negative-report region collapses under tuning | **holds, but the mechanism is open.** Neutral Mask suggests probing the representation; we only measured the readout. |
| the tuned model has no inducible negative state at this band | **at risk.** Section 1. Needs the depth sweep before it is written down. |

## Immediate next steps this lit check generates

1. **Depth sweep for the negative pole**, layers spanning 14-27% and 53-66% of depth on
   Qwen2.5-3B-Instruct and its base sibling. This is now the highest-value experiment in the project
   and it is cheap. Preregister before running.
2. Read arXiv:2604.04064 on base-versus-instruct emotion extraction before choosing how to fit
   directions in that sweep.
3. Read the Neutral Mask paper in full and decide whether a probe on the tuned model's residual
   stream is in scope.
4. Re-read Venkatesh section 4.4 and Appendix B for the domain-level controls, which are a better
   model of confound control than what we did on arm C.
