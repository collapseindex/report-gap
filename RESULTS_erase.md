# Results: the state survives erasing the vector that caused it

**Run 2026-08-01.** 10560 rows, artifacts committed unscored before any endpoint was computed.
Scored against `PREREG_erase.md`.

**Verdict: TRANSFORMED**, on two of four erase layers, with the confound this design cannot fully
escape stated in full below.

This arm answers the objection `RESULTS_shell.md` raised against itself. It does **not** rehabilitate
that document's headline, which the replication killed. What it rehabilitates is the narrower
representational half.

---

## The test

Orthogonalizing a probe against the injected direction removes that vector from the *measurement*.
It does not remove it from the *stream*, so the probe may be reading the wake of what we pushed.
This removes it from the stream: inject `v` at layer 24, project the `v` component **out** of the
residual at layer `E`, read the orthogonalized probe at layer 32.

## Instruct model

| erase at | erase_only vs baseline | capability (pos) | **primary (neg vs random)** | gate |
|---|---|---|---|---|
| 25 | +0.1474 | +0.5219 | -0.3971 | **FAILED** |
| **26** | +0.0976 | +0.8844 | **-0.5933** [-0.6260, -0.5604] | ok |
| 28 | +0.1139 | +0.9249 | -0.6844 | **FAILED** |
| **30** | +0.0408 | +1.2099 | **-0.8532** [-0.8875, -0.8188] | ok |

All in standard deviations of the baseline probe score. Without any erase, the same contrast is
**-0.9909**.

So at layer 30, **86% of the effect survives projecting the injected direction out of the stream**
(-0.8532 against -0.9909). At layer 26, 60% survives. Both Holm-corrected across erase layers.

**Two of four erase layers are invalidated by their own gate.** Projecting a direction out is a
perturbation, and at layers 25 and 28 it moves the probe by more than the 0.10 SD floor with no
injection present. Their primaries are `uninformative` and are not interpreted, including in the
profile.

## The confound, and why it does not account for the result

Erasing earlier perturbs more: the `erase_only` artifact falls from +0.1474 at layer 25 to +0.0408
at layer 30. So a primary that grows with erase depth is partly just a shrinking perturbation, and
the preregistered profile clause cannot separate those on its own.

The diagnostic that can, **computed after seeing the data and labelled exploratory**: if the profile
were purely the erase perturbing less at later layers, the primary and the artifact would scale
together and their ratio would be flat.

| erase at | \|primary\| | erase_only | ratio |
|---|---|---|---|
| 25 | 0.3971 | 0.1474 | 2.7 *(gate failed)* |
| **26** | 0.5933 | 0.0976 | **6.1** |
| 28 | 0.6844 | 0.1139 | 6.0 *(gate failed)* |
| **30** | 0.8532 | 0.0408 | **20.9** |

It is not flat. The surviving signal grows roughly eightfold relative to the perturbation between
the two gate-clean layers. A pure perturbation account does not predict that.

## What this licenses

> After projecting the injected direction out of the residual stream, a probe orthogonal to that
> direction still reads most of the injected state, and more of it the later the projection is
> applied, while the same projection without injection barely moves the probe.

That is the mechanical content of "the model transformed the injection into something not along it",
and it is the claim `RESULTS_shell.md` could not make because orthogonality alone does not rule out
linear persistence.

## What this does NOT license

- **It does not restore the SHELL verdict.** `RESULTS_replication.md` showed the option readout does
  move at other orderings, so the dissociation between representation and expression is dead. This
  arm is about the representation half only, and it deliberately routes nowhere near the option
  channel: option mass was recorded here and carries no verdict by preregistration.
- **Not experience, welfare, or concealment.** A probe reading a correlate after a projection is a
  statement about representation geometry. Nothing here is about what the model feels or withholds.
- **Not a claim that the transformation is meaningful computation.** "Not along `v`" is a
  geometric fact. Whether the model's own circuits read the transformed quantity for anything is
  untested, and the one readout we checked does not obviously use it.

## Caveats, in order of how much they should worry a reader

1. **The profile rests on two points.** Half the erase layers were invalidated by their own gate, so
   the preregistered temporal profile is a two-point comparison. Two points make a line, not a
   profile.
2. **The clean layers are not equally clean.** Layer 26's erase artifact is +0.0976 against a 0.10
   floor. It passes by 0.0024. Layer 30 at +0.0408 is the only comfortable one.
3. **The ratio diagnostic is post hoc.** It was computed after seeing that the preregistered profile
   was confounded. It is reported as exploratory and the verdict does not formally depend on it,
   though the honest reading does.
4. **Between-ordering variance is 85% of within-ordering variance** for the baseline probe (0.4897
   against 0.5749). Every endpoint here is paired per cell so ordering differences out, but the
   readout is pervasively ordering-sensitive and this is the arm where that was finally measured
   first rather than last, per the requirement added after the replication failure.
5. **`p_orth` and `d_hat` are fit from the same lexical contrast set** at different layers. A
   subspace shared between them that the erase does not reach remains possible. The design that
   would rule it out is a probe fit from an independent contrast set, named exploratory in the
   prereg and not run.
6. One model pair, one injection layer, one alpha, m = 2 random battery.
