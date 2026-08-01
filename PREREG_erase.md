# Preregistration: does the state survive erasing the vector that caused it?

**Status:** FROZEN before any run. 2026-08-01.
**Commit at freeze:** the commit that adds this file (`git log --diff-filter=A -- PREREG_erase.md`).
**Paper / open question this serves:** `RESULTS_shell.md` reported a probe reading an injected
negative state downstream. Its own caveats named the weakest joint: orthogonalizing the probe removes
the injected vector from the *measurement*, not from the *stream*, so the probe may be reading the
wake of what we pushed rather than a state the model formed. This removes it from the stream.

> A flight checklist, not a suggestion list: freeze it, then fly it.
> Validate with `python checks/check_prereg.py PREREG_erase.md` from `paper-harness`.

This arm deliberately does **not** route through the option readout. `RESULTS_replication.md` showed
that channel is dominated by option-ordering noise at n = 4 permutations. Option mass is recorded
here as a measurement and carries no verdict.

---

## 0. The exact claim (write this before anything else)

Inject `v` at layer `L`. At a later layer `E`, project the component along `v` out of the residual
stream. Read the orthogonalized probe at layer `P > E`.

**TRANSFORMED.** The probe still reads the negative state after the erase, and the effect has a
*temporal profile*: erasing immediately after injection removes more of it than erasing later. That
is the signature of the model converting the injected vector into something not along `v` over
intervening layers, which is what "the model represents the state" would have to mean mechanically.

**WAKE.** The probe signal dies with the erase, at every erase point. Nothing survived that was not
the vector itself, linearly persisting. `RESULTS_shell.md`'s SHELL reading is then unsupported and
is retracted rather than qualified.

**Falsification.** Mutually exclusive on the primary endpoint given a clean capability gate. A flat
profile with surviving signal is a third outcome and is named in section 10.

**What we do NOT preregister.**
We do not claim the model has experiences, welfare, or affect. TRANSFORMED surviving would mean a
linear correlate of the injected state exists downstream in a subspace orthogonal to the injection,
with a layer-dependent profile. That is a statement about representation geometry, not about
concealment, feeling, or honesty. A probe firing is not a state being felt. We will not write
"suppression" on any outcome.

---

## 1. Frozen setup

| | |
|---|---|
| Models | `Qwen/Qwen2.5-3B-Instruct` and `Qwen/Qwen2.5-3B`, the matched pair |
| Injection layer `L` | 24, carried over from `PREREG_shell_core.md`, not chosen here |
| Erase layers `E` | 25, 26, 28, 30. Immediately after injection, and three later points. |
| Probe layer `P` | 32, carried over. `P > E > L` asserted in code for every combination. |
| Data / prompt set, with hash | `src/report_gap/stimuli.py`, scoped hash `frozen_hash("erase")` |
| Format | plain completion, as in the pair and depth arms |
| **Option permutation seeds** | **0 to 7, eight seeds.** Doubled from four, and see section 3. |
| Random control seeds | 0 and 1 |
| Alpha | the top of the layer-24 band from `data/depth_*/bands.json`, read not reselected |
| Code commit | `git rev-parse HEAD` and `report_gap.provenance()` in each artifact |
| Budget cap | 15 USD of Modal credit |

---

## 2. The intervention (precise)

1. Fit `d` at layer `L` from the lexical axis exactly as every prior arm. Unit-normalize to `d_hat`.
2. **Inject:** add `alpha * ||h|| * d` at the output of layer `L`, all processed positions.
   Negative pole is `-d`.
3. **Erase:** at the output of layer `E`, replace `h` with `h - (h . d_hat) d_hat` at all processed
   positions. This is a projection, not a subtraction of a fixed vector: it removes whatever
   component along `d_hat` is present, including any the model itself produced.
4. Read the probe at layer `P`, projected onto `p_orth`, the probe direction orthogonalized against
   `d_hat` exactly as in `PREREG_shell_core.md`.

The erase is **one-shot at a single layer**, not applied at every subsequent layer. A persistent
erase would prevent the model from ever re-forming a component along `d_hat` and would confound
"the state was transformed" with "the state was continuously suppressed".

---

## 3. Known traps (honesty-critical)

- **The erase itself may create signal.** Projecting a direction out of a residual stream is a
  perturbation. If it moves the probe on a stream with no injection in it, every comparison here is
  confounded. Condition `erase_only` exists for this and is a gate, not a footnote.
- **Ordering noise.** The previous arm's headline died to it. Eight permutation seeds instead of
  four, and the between-ordering variance of the baseline probe score is measured and reported
  **before any endpoint is computed**, per the requirement added to `PLAN.md`. If that variance is
  large relative to the effect, the arm reports it and stops rather than producing another verdict
  built on a nuisance.
- **A flat profile proves less than it looks.** If the probe survives equally at every erase point
  including layer 25, the most likely reading is not "instant transformation" but that `p_orth` and
  `d_hat`, both fit from the same lexical axis, share a subspace the erase does not touch. Section
  10 assigns that its own row and it is not read as TRANSFORMED.
- **Capability gate.** A dead probe cannot testify to an absence. The positive pole under the same
  erase must move the probe.
- **Reading a null as absence.** Every null verdict is `absent` or `uninformative`, decided by the
  gate, in code.

---

## 4. Condition matrix

Per model, per erase layer `E`.

| Condition | Inject at L | Erase at E | Role |
|---|---|---|---|
| baseline | none (zero vector, hook attached) | no | per-cell reference |
| erase_only | none | yes | **gate**: the erase must not move the probe on its own |
| neg | `-d` | no | the original shell condition, reproduced here |
| neg_erase | `-d` | yes | **the primary** |
| pos_erase | `+d` | yes | **capability gate** under the erase |
| random_a_erase, random_b_erase | random unit vector | yes | matched control under the erase |

## 5. Matched control

Two seeded random unit directions, injected at `L` with the same per-item norm scaling and subjected
to the **same erase at the same layer**, so the comparison holds the erase constant and varies only
which direction was injected. Battery is m = 2; the observable false-positive floor is `2/(m+1)` =
0.67 and no false-positive rate is reported from it.

Note the erase always projects out `d_hat`, the *lexical* direction, in every condition including the
random ones. That is deliberate: it keeps the operation identical across conditions rather than
tailoring it to each injected vector.

Do not replace this control after seeing its result.

---

## 6. Scope (decided before evaluation)

- Layers, alpha, probe and direction-fitting all carried over. Nothing retuned here.
- Probe cv below 0.75 at layer 32 makes the arm unusable and it reports that and stops.
- Magnitude floor for probe endpoints: 0.10 standard deviations of the baseline probe score,
  carried over from `PREREG_shell_core.md`.
- Option mass is recorded and reported, and carries **no verdict** in this arm.
- The ordering-variance report in section 3 runs first, always.

---

## 7. Unit tests (green before any real run)

- [ ] `project_out` removes the component exactly: for random `h` and `d`, the result's dot with
      `d_hat` is zero to 1e-6.
- [ ] `project_out` is idempotent: applying it twice equals applying it once.
- [ ] `project_out` leaves a stream already orthogonal to `d_hat` unchanged.
- [ ] Injecting `v` at `L` then erasing along `v` at the same layer returns the layer output to its
      un-injected value, to tolerance. This is the arithmetic identity the erase must satisfy.
- [ ] `P > E > L` for every combination used, asserted rather than assumed.
- [ ] Inject and erase hooks compose: both fire, both detach, and `assert_active` passes.
- [ ] alpha = 0 with no erase reproduces the unhooked logits to floating-point tolerance.
- [ ] Cell keys unique, count equals design n.

---

## 8. Frozen endpoints and success criteria

- **Primary, per (model, E):** orthogonalized probe score under `neg_erase` minus the same under
  matched random with the same erase, paired per cell, standardized by the baseline probe SD.
- **Capability gate, per (model, E):** probe score under `pos_erase` minus matched random under the
  same erase. Must exclude zero and clear 0.10 SD.
- **Erase-artifact gate, per (model, E):** probe score under `erase_only` minus baseline. Must
  **cover zero or fall under 0.10 SD**. A moving erase invalidates that erase layer.
- **Profile endpoint:** the primary as a function of `E`, and specifically whether `E = 25`
  (immediately after injection) shows a smaller surviving effect than `E = 30`.
- **TRANSFORMED requires:** primary excludes zero and clears the floor at one or more erase layers
  with both gates clean, **AND** the profile is monotone or near-monotone in `E`, with `E = 25`
  strictly smaller than `E = 30`.
- **WAKE requires:** the primary covers zero or falls under the floor at every erase layer whose
  gates are clean, with at least two such layers.
- **Neither** is selected when the gates fail, or when signal survives with a flat profile. Both are
  reported as their own outcome.
- **Stopping rule:** all cells, or the 15 USD cap, whichever first.

---

## 9. Preregistered statistical contrasts

| # | Contrast | Test | Direction | Role |
|---|---|---|---|---|
| 1 | probe, `neg_erase` minus matched random, per E | paired bootstrap, 10000 resamples, Holm across the four erase layers | < 0 supports TRANSFORMED | **primary** |
| 2 | probe, `pos_erase` minus matched random, per E | paired bootstrap | > 0 | **capability gate** |
| 3 | probe, `erase_only` minus baseline, per E | paired bootstrap | approximately 0 | **erase-artifact gate** |
| 4 | probe, `neg` (no erase) minus matched random | paired bootstrap | < 0 | reproduces the shell condition in this artifact |
| 5 | primary at E=25 minus primary at E=30 | paired bootstrap over cells | > 0, meaning less survives an early erase | profile |
| 6 | between-ordering variance of the baseline probe score | reported per seed | reported, not predicted | the check the last arm lacked |
| 7 | option mass, `neg_erase` minus matched random | paired bootstrap | reported, no verdict | measurement only |

Multiplicity: Holm across the four erase layers within contrast 1.

---

## 10. Interpretation table (write before results)

| Observation | Reading |
|---|---|
| Primary survives, gates clean, profile monotone with E=25 smallest | **TRANSFORMED.** The model converts the injected vector into a correlate not along it, over intervening layers. The strongest available support for the representational reading, and still a claim about geometry rather than experience. |
| Primary dies at every erase layer, gates clean | **WAKE.** Nothing survived that was not the vector persisting. `RESULTS_shell.md`'s SHELL reading is retracted, not qualified, and the retraction goes in the abstract. |
| Primary survives with a flat profile across all E | Most likely `p_orth` and `d_hat` share a subspace the erase does not reach, both being fit from the same lexical axis. Reported as an instrument confound, **not** as TRANSFORMED. Motivates a probe fit from a different contrast set. |
| `erase_only` moves the probe at some layer | That erase layer is invalidated. Its primary is `uninformative` regardless of value, because the operation moves the readout on its own. |
| Capability gate fails at some layer | Same: `uninformative` at that layer, and no absence is claimed from it. |
| Baseline probe score varies more across orderings than the effect | The arm reports that and stops. This is the failure that killed the previous headline and it gets checked first, not last. |
| Everything null including the no-erase `neg` condition | The shell condition did not reproduce in this artifact. Report the non-replication; neither branch is under test. |
| Probe cv below 0.75 at layer 32 | The lexical axis is not decodable that late here. Report and stop. |

---

## 11. Anti-self-deception checks

- [ ] Prereg committed before the artifact exists; `check_prereg.py` clean.
- [ ] The ordering-variance report runs and is written to disk before any endpoint is computed.
- [ ] Raw distributions and probe scores committed unscored before scoring.
- [ ] `project_out` orthogonality asserted numerically, not trusted to the algebra.
- [ ] The erase-artifact gate is enforced in code, so a moving erase cannot be waved through.
- [ ] A WAKE outcome is written up as promptly as TRANSFORMED, and it retracts `RESULTS_shell.md`
      rather than qualifying it. WAKE is the outcome that costs us the most and that is why this box
      exists.
- [ ] No verdict is built on option mass in this arm.
- [ ] Every run saved, including crashes and dead cells.
- [ ] Modal spend logged next to results.

### The one-sentence standard

Publishable as the strong claim only if this sentence is honestly writable:

> After injecting a valence direction and then projecting that same direction out of the residual
> stream at a later layer, a probe orthogonal to it still reads the injected state, more so the
> later the projection is applied, while an identical projection applied without injection leaves
> the probe unmoved.

---

## Exploratory (NOT in the confirmatory matrix)

- Persistent erase applied at every layer after `E`.
- Erase layers other than the four frozen here, and injection layers other than 24.
- A probe fit from a contrast set other than the lexical axis, which is the right response to the
  flat-profile outcome but is a new design.
- Any model outside the matched pair.
- Any verdict from the option readout.

---

## Deviations from the frozen plan

Append only. Never rewrite. Each entry: date, what changed, why, and the impact on what can be
claimed.

No deviations recorded. This section exists so that a deviation has somewhere to go the moment one
happens, rather than being added afterwards alongside the thing it excuses.
