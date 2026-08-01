# Results: the arm cannot answer its own question, and that weakens our surviving claim

**Prereg:** [PREREG_prompt_erase.md](PREREG_prompt_erase.md), frozen before the runner existed.
**Verdict:** **ERASURE_UNINFORMATIVE.** The primary question is unanswerable as designed.
**Artifacts:** `data/pe_*/`, 4,720 rows, no injection anywhere. **Compute:** ~2 min A100.

---

## The gate that failed, and why it matters that it is a gate

The arm's question was: erase the valence subspace at layer `E`, **verify the state is no longer
decodable there**, then ask whether it is decodable again at layer 32. Step two is the whole design.

**Step one never happened.** A probe refit on the erased activations at layer `E` separates aversive
from pleasant at **cv 1.000 at every k, on both models, at both layers, up to k=56 with n=60
samples.** The property is not removed. Per the preregistered interpretation table this is
`report and stop`, and no re-encoding claim is made.

The reason is structural rather than a bug: with 60 samples in a 2048-dimensional residual stream, a
cross-validated linear probe separates the classes after almost any rank-56 subspace is removed.
**The erasure check cannot fail at this sample size**, which means the design needs far more items
than 30 topics per framing before it can ask its question at all. That is the honest limit and it
was found by the gate rather than by inspection.

---

## What is descriptively true anyway, with no gate riding on it

The specificity control comes back clean, which is worth reporting even though the primary is dead.
Layer-32 separation between aversive and pleasant contexts, as a fraction of the un-erased value:

| erase layer | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=56 | random k (any) |
|---|---|---|---|---|---|---|---|---|
| **L30, base** | 60% | 43% | 35% | **31%** | 34% | 32% | **30%** | **~100%** |
| **L26, base** | 85% | 78% | 71% | 66% | | | | ~97% |
| **L30, instruct** | 66% | 52% | 44% | 39% | | | | ~99% |

Two things are solid here:

1. **A random subspace of the same rank removes essentially nothing** (~100% of the separation
   survives at every k). So the reduction under the fitted basis is about *which* directions were
   removed, not about removing dimensions.
2. **It plateaus at about 30%.** Going from k=8 to k=56, a sevenfold increase in erased dimensions,
   removes no further signal. Roughly a third of the layer-32 separation lives in a subspace the
   fitted directions never reach.

That plateau is suggestive of exactly the re-encoding the arm was built to test. **We are not
claiming it**, because the erasure gate failed and we therefore never established the state was
absent at `E` in the first place. A suggestive plateau behind a failed gate is `uninformative`, and
the whole point of putting the gate in code was to stop us reading it as more.

---

## The part that costs us: this weakens `RESULTS_erase.md`

`RESULTS_erase.md` reports that **86% of an injected state survives projecting the injected direction
out of the stream**, and that is the one substantive claim this project still has standing on its
own. This arm makes the operation behind that number look much weaker than the word "erase"
suggests.

Here, on a comparable readout, **removing one fitted direction removes 40% of the signal at L30 and
15% at L26**. Removing eight removes 69% and 34%. The single-direction projection is not a small
perturbation of a full erasure; it is a small fraction of one, and it never made the property
undecodable at all.

So the honest reading of the erase result is narrower than it was: **86% of the effect survives an
operation that removes one direction and demonstrably does not remove the property.** That is a
weaker statement than "the state outlives its own cause", and the paper now says the weaker one.

The arms are not identical (injected vs prompt-induced state, different probe, different contrast),
so this is an argument by analogy about the *operation*, not a direct re-measurement of that number.
The direct version would be to re-run the injected erase arm at k>1, which we have not done.

---

## Deviation logged

The frozen matrix was `k` in {0,1,2,4,8}. After every frozen `k` returned refit cv 1.000, we added
`k` in {16,32,48,56} to find where erasure bites at all. Those rows are marked `[exploratory k]` in
the analyzer output and in the table above. **The frozen conclusion does not depend on them**: the
gate had already failed at every preregistered `k`, and the exploratory rows only confirm it fails
up to the point where `k` approaches the sample size.

Also logged: the runner initially read the layer-32 probe at `hidden_states[PROBE_LAYER]` where the
probe had been fit through `collect_activations`, which uses `[layer + 1]`. That mismatch applied a
probe fit on layer 32's output to layer 31's output. It was caught because the killer control
returned *identical* numbers for the fitted and random bases, which is not something a real effect
does. Fixed before the reported run. **The project's other arms were already correct**:
`collect_activations` and `modal_erase.py` both use `[layer + 1]`, so no existing result is affected.

---

## What this arm does NOT show

- It does not show the state is re-encoded. The gate that would license that failed.
- It does not show the state is *not* re-encoded either. A failed gate is not a null.
- Prompt-induced valence is confounded with prompt content by construction, as the prereg said in
  advance: the aversive context says the document is bad, and a probe separating the framings may be
  reading that rather than any state of the model. The clean separation of 2.36 SD is consistent with
  a probe reading obvious lexical differences.
- Nothing here bears on whether models have experiences, welfare, or affect.

## What would fix it

More items. The erasure check needs `n` large enough that a linear probe cannot separate the classes
by chance in the residual dimensions, which means hundreds of contexts rather than 30. That is cheap
to generate and cheap to run, and it is the obvious next version of this arm.
