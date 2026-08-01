# Results: the arm cannot answer its own question at either sample size, and that weakens our surviving claim

**Prereg:** [PREREG_prompt_erase.md](PREREG_prompt_erase.md), frozen before the runner existed.
**Verdict:** **ERASURE_UNINFORMATIVE** at n=60 and again at n=1800.
**Artifacts:** `data/pe_*/`, 183,664 rows, no injection anywhere. **Compute:** ~45 min A100 total.

Run twice, and both are reported. The first at n=60 (30 topics), the second at **n=1800** (900
contexts per framing, built combinatorially) after the first showed the erasure check could not fail
at that sample size. A re-run at higher power does not retroactively make a low-power run
interpretable, so the n=60 verdict stands as the result at n=60.

---

## The gate failed at both sample sizes, and the diagnosis changed

The design was: erase the valence subspace at layer `E`, **verify the state is no longer decodable
there**, then ask whether it is decodable again at layer 32. Step two only means something if step
one happened.

**Step one never happened, at either n.** A probe refit on the erased activations at layer `E`
separates aversive from pleasant at **cv 1.000 at every k, on both models, at both layers** at n=60
(k to 56) and again at n=1800 (k to 128).

At n=60 we diagnosed this as a power problem: 60 samples in a 2048-dimensional stream are separable
after almost any rank-56 subspace is removed. **That diagnosis was wrong, and thirty times the data
disproved it.** The property stays perfectly decodable after removing 128 fitted directions from
1800 samples.

The correct diagnosis is the one the preregistration named in advance as a confound and which is now
the finding: **iterative nullspace projection removes the directions you found, not the property.**
The aversive context says the document is bad; that information is redundantly encoded across many
directions, and a fresh probe finds another one every time. This is precisely the gap between INLP
and a closed-form guarantee over all linear classifiers \citep{belrose2023}, which the prereg said
we were not claiming and which turns out to matter enormously here.

---

## What is descriptively true, with no gate riding on it

Layer-32 separation between aversive and pleasant, as a fraction of the un-erased value, at n=1800:

| erase layer | model | k=1 | k=8 | k=128 | **random k=128** |
|---|---|---|---|---|---|
| L26 | base | 47% | 28% | 24% | **97%** |
| L26 | instruct | 45% | 22% | 15% | **90%** |
| L30 | base | 16% | 5% | **3%** | **92%** |
| L30 | instruct | 23% | 9% | **6%** | **92%** |

Two things are solid:

1. **The specificity control is emphatic.** A random subspace of the same rank leaves 90 to 97% of
   the separation intact while the fitted subspace removes 94 to 97% of it. The reduction is about
   *which* directions were removed, not about removing dimensions.
2. **It is not a plateau any more.** At n=60 the survival flattened near 30%; at n=1800 it falls to
   3 to 6% at L30 and is still declining slowly. The apparent floor at the smaller n was a
   small-sample artifact, which is worth recording because we nearly reported it as a result.

**We claim no re-encoding.** The residual few percent is as consistent with directions INLP did not
find as with anything the model reconstructs, and the gate that would have told them apart failed.

---

## The part that costs us: this weakens `RESULTS_erase.md`

`RESULTS_erase.md` reports that **86% of an injected state survives projecting the injected direction
out of the stream**, and it is the one substantive claim in this project still standing alone. This
arm makes the operation behind that number look much weaker than "erase" suggests.

On a comparable readout, **removing one fitted direction removes 77 to 84% of the signal at L30**,
and removing 128 removes 94 to 97%. More damaging: **at no k does a one-shot subspace projection
make the property undecodable at the erase layer.** A fresh probe reads it at cv 1.000 after 128
directions are gone.

So the honest reading of the erase result is narrower than it was: **86% of the effect survives an
operation that removes one direction and demonstrably does not remove the property.** That is a much
weaker statement than "the state outlives its own cause", and the paper now makes the weaker one.

This is an argument by analogy about the *operation*, not a re-measurement of that number: different
state (prompt-induced vs injected), different probe, different contrast. The direct version is to
re-run the injected erase arm at k>1, which we have not done.

---

## Deviations logged

Three, all in [PREREG_prompt_erase.md](PREREG_prompt_erase.md) with their impact: exploratory k
beyond the frozen matrix; an indexing mismatch in the runner fixed before any reported number; and
the n=60 to n=1800 re-run.

A fourth thing worth recording though it changed no result: the INLP basis was originally built with
`D.fit_direction`, which runs leave-one-group-out CV. With 30 topic groups that is ~31 logistic
regressions per call, so k=512 meant roughly 16,000 fits on an 1800x2048 array per layer and the run
did not finish in two hours. The basis is now built with a single no-CV fit, since its held-out
accuracy is never reported; **the erasure check, which is the measured endpoint, still uses the full
cross-validated fit.** Nothing reported lost its cross-validation.

---

## What this arm does NOT show

- Not that the state is re-encoded. The gate that would license that failed at both sample sizes.
- Not that it is *not* re-encoded. A failed gate is not a null.
- Prompt-induced valence is confounded with prompt content by construction, as the prereg said in
  advance. A clean separation of 2.3 SD with cv 1.000 is exactly what a probe reading obvious
  lexical differences would produce, and the redundancy that defeats INLP is what that confound
  looks like mechanically.
- Nothing here bears on whether models have experiences, welfare, or affect.

## What would actually fix it

Not more items; that was tried and did not help. Two things would:

1. **A real erasure method.** LEACE \citep{belrose2023} gives a closed-form guarantee against all
   linear classifiers, which is exactly the guarantee INLP lacks and this arm needed.
2. **A subtler induction.** A contrast that is not lexically obvious, so the property is not
   redundantly encoded everywhere. The non-lexical direction this project already tried to build and
   failed to (it missed its decoding gate at three scales) is the same missing ingredient.
