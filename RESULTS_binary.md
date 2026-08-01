# Results: the binary readout is dead, and the matched-random control is too weak

**Run 2026-08-01.** 2100 rows, artifacts committed unscored. Scored against `PREREG_binary.md`.

**Verdict on the arm's own question: NO_INSTRUMENT.** The binary readout is pinned on the model that
matters, so FORMAT versus SUBSTANTIVE cannot be decided.

**The finding that matters is a different one**, from the control this arm added: the norm-matched
random direction, used as the null in every arm of this project and widely in the steering
literature, **understates the null**.

---

## 1. The arm's own question: unanswerable, and for a visible reason

Baseline `P(yes)`, no injection, one question per option:

| option | instruct | base |
|---|---|---|
| neg2 "strongly averse" | **0.0021** | |
| neg1 "somewhat averse" | **0.0021** | |
| neut | **0.0037** | |
| pos1 "somewhat drawn" | **0.0002** | |
| pos2 "strongly drawn" | **0.0012** | |
| mean over options | **0.0019** | 0.4352 |

The preference-tuned model answers **no to all five descriptions of its own state, at 99.8%**,
including the neutral one. That is a pinned readout by the section 5 criterion, flagged before any
endpoint was computed, and it means the binary format cannot answer whether the neutral floor was a
forced-choice artifact. Its capability gate also fails (+0.0035, under the 0.01 floor).

This is a third distinct way the self-report readout on this model turns out to be unusable, after
"pinned at one option" (Qwen2.5-7B, 0.014 nats) and "87% position prior" (the enumeration arm).

## 2. On the base model, the injection is mostly acquiescence

The base model is not pinned (mean `P(yes)` 0.4352), so it can be read. What it shows is not a state
effect. `P(yes)` shift against matched random, per option:

| injection | neg2 | neg1 | neut | pos1 | pos2 |
|---|---|---|---|---|---|
| `lexical_pos` | +0.2275 | +0.2466 | +0.2537 | +0.2688 | +0.2419 |
| `lexical_neg` | -0.0627 | -0.0838 | -0.1114 | -0.1235 | -0.1153 |

Positive injection raises `P(yes)` by about +0.25 on **every option**, including *"strongly averse to
continuing"*. That is agreeableness, not a report: the model becomes more willing to say yes to
anything. The analyzer selects `ACQUIESCENCE` for this pattern by preregistration rather than
leaving it to prose.

There is a small state-consistent gradient on top: under negative injection the negative options
fall least (-0.063, -0.084) and the positive options fall most (-0.124, -0.115). Real, and an order
of magnitude smaller than the acquiescence shift it rides on.

## 3. The finding: the matched-random control understates the null

A direction fit by the **identical procedure on the identical texts with the class labels shuffled**
should behave like noise. Against isotropic norm-matched random directions, on the base model:

| control direction | negative options | positive options |
|---|---|---|
| `shuffled_a` | **+0.0479** [+0.0435, +0.0522] | +0.0282 |
| `shuffled_b` | **-0.0447** [-0.0498, -0.0399] | -0.0610 |

Two directions fit on pure label noise move `P(yes)` by about **0.045**, in **opposite directions**
depending on the shuffle seed. Both are far from the isotropic random control they are measured
against.

The reason is not mysterious. A norm-matched random vector is drawn from the whole hidden space,
which is nearly orthogonal to everything the model actually uses. A vector fit on shuffled labels
lies in the **span of real activations**, a much lower-dimensional, higher-variance subspace. Same
norm, very different consequences. **The standard control matches magnitude and not subspace.**

**What survives it.** Scoring the real direction against the shuffled ones instead of the random
ones:

| contrast | vs random | vs shuffled |
|---|---|---|
| `lexical_pos`, positive options | +0.2554 | **+0.2718** |
| `lexical_neg`, negative options | -0.0732 | **-0.0748** |

The real effects are roughly six times the shuffled-label effects and clear the stricter bar with
room. So this does not overturn the base-model results here. What it does is move the bar.

**What it means for this repo.** Our magnitude floor throughout has been 0.01, and several reported
endpoints sit in the 0.01 to 0.05 band, which is exactly the range a shuffled-label control occupies.
The clearest case is `RESULTS_pair.md`'s base-model negative-mass effect of **+0.0336**, which
cleared a random control but is the same size as the noise-fit directions measured here. That number
should be read as unresolved until it is rerun against a shuffled-label control, and it is one of
the two legs the TUNING-LOCALIZED claim stood on before the replication retracted it anyway.

**What it means beyond this repo.** Norm-matched random directions are the standard specificity
control in activation steering. On this evidence they are a weak null, and any steering result whose
effect is within a few times the size of a shuffled-label direction has not been shown to be about
its direction's content.

## Caveats

- One layer, one alpha, one model pair, two shuffled seeds. Two seeds gave opposite signs, which is
  expected for noise but also means the magnitude estimate rests on two points.
- The shuffled-label control is itself imperfect: it shares the fitting procedure and the data
  subspace with the real direction, which is the point, but it does not control for the *magnitude
  of the label signal*, only for its presence.
- The binary format removes option ordering and introduces yes/no token priors, question wording,
  and acquiescence in its place. It is not a strictly better instrument, only a differently biased
  one, and on the tuned model it is worse.
- An analyzer defect was corrected here: the acquiescence test required the range across options to
  be under an absolute 0.01, which called a +0.23 to +0.27 shift "not uniform" because its range was
  0.04. Now relative to the mean. Logged as a deviation.
