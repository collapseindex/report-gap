# Results: LEACE erases most of it, not all of it, and the previous arms' gate was reading the wrong tensor

**Prereg:** [PREREG_leace.md](PREREG_leace.md), frozen before the runner existed.
**Verdict:** **ERASURE_UNINFORMATIVE** on both models, for the first time for a real reason.
**Artifacts:** `data/leace_*/`, 10,812 rows, no injection. **Compute:** ~2.5 min A100.

---

## The bug this arm found, which invalidates a measurement in two earlier arms

**The erasure check was reading upstream of the hook.** A forward hook on layer `E` first changes
`hidden_states[E+2]`; `hidden_states[E]` and `[E+1]` are untouched. Every previous erasure check read
`[E+1]`, i.e. **un-erased activations**, which is why it returned cv 1.000 no matter what was erased.
Verified directly: a hook that **zeroes the entire stream** at layer 26 leaves `hidden_states[26]`
and `[27]` bit-identical and first moves `[28]` by 95.2.

That gate could not fail. It was not measuring erasure at all.

Consequences, stated plainly:

- `RESULTS_prompt_erase.md`'s cv=1.000, at n=60 and n=1800, was this bug. The conclusion drawn from
  it, "INLP removes the directions you found, not the property," is **withdrawn for a second and
  more basic reason** than the protocol bias recorded there yesterday. Neither run measured erasure.
- The **layer-32 results in every arm are unaffected.** They read at index 33, which is downstream of
  hooks at layers 25-30, which is why they always moved. The narrowing of `RESULTS_erase.md` rests on
  those and still stands.
- The runner now asserts the hook reaches the index the check reads, by zeroing the layer and
  requiring the read to move. A gate that cannot fail is worse than no gate.

---

## With the check pointed at the right tensor

Eraser fit on 20 train topics, every number below read on 10 **held-out** topics, n=300.

| model | layer | held-out decodability at `E` | held-out class-mean gap | layer-32 survives | random rank-1 |
|---|---|---|---|---|---|
| base | 26 | 1.000 -> **0.993** | 26% of clean | **32%** | 100% |
| base | 30 | 0.997 -> **0.720** | 18% of clean | **5%** | 100% |
| instruct | 26 | 1.000 -> **1.000** | 27% of clean | **30%** | 100% |
| instruct | 30 | 1.000 -> **0.748** | 18% of clean | **9%** | 100% |

**LEACE does erase, substantially and specifically.** The class-mean gap falls to 18-27% of clean on
held-out items, and at layer 30 the residual decodability falls from ~1.000 to 0.72-0.75. A
**rank-matched random** eraser leaves 100% of the layer-32 separation intact while rank-one LEACE
removes 68% at L26 and 91-95% at L30. One dimension, correctly chosen, does almost all of it.

**But the gate still fails at the preregistered 0.60 bar**, so the primary question stays
unanswerable and the verdict is `ERASURE_UNINFORMATIVE`. The difference from the previous arms is
that this is now a real measurement of incomplete erasure rather than an artifact of reading the
wrong tensor.

**Why erasure is incomplete, and it is not a bug.** The eraser is fit on train topics and applied to
held-out topics. LEACE's guarantee is that the class-conditional means coincide *on the distribution
it was fit to*; the held-out topics are a shifted distribution, so a train-fit rank-1 eraser does not
fully collapse them. That is a real generalization gap, and it is the honest thing to measure: an
eraser fit on the same rows it is evaluated on would confirm itself, which is the mistake this arm
was built to avoid.

---

## What this arm does and does not license

- It does **not** show re-encoding. The gate failed; a failed gate is not a null.
- It does **not** show that the state fails to survive erasure either.
- It **does** show that a rank-one, correctly-chosen direction accounts for the great majority of a
  prompt-induced valence signal at layer 32, against a rank-matched random control that accounts for
  none of it. That is a specificity result that survives the gate failure because it never depended
  on it.
- It **does** show that three previous conclusions rested on a gate that could not fail.
- Prompt-induced valence remains lexically confounded, unchanged.

## What would fix it, now specifically

Fit and evaluate the eraser on the **same distribution** while keeping the probe honest: split by
topic for the *probe* but fit the eraser on all topics, and report both the self-fit gate (which
LEACE guarantees) and the transfer gate (which it does not). The two numbers answer different
questions and this arm conflated them into one bar.
