# Changelog

Semantic versioning. Dates are absolute.

## v0.4.0 (2026-08-01)

Five preregistered arms run end to end on Modal. Every raw artifact committed unscored before its
endpoints were computed.

**Added**
- `PREREG_base_pair.md`, `PREREG_depth.md`, `PREREG_shell_core.md` and their runners, scorers and
  results.
- `RELATED_WORK.md`, a lit check with read-depth marked per source. It found a live threat to the
  central null (arXiv:2605.05653 on injection depth), which was then tested and survived.
- `planted.py`, the planted-discrepancy control: a synthetic effect of known size in the exact
  statistic the decision rule reads, at full strength and at the claimed detection floor.
- `analysis.is_dead` and `analysis.is_saturated`, which distinguish a readout pinned before the
  injection from one pinned by it. Standard integrity checks cannot see either.
- Probe orthogonalization against the injected direction, with the residual asserted at 1e-6 and
  the un-orthogonalized number reported alongside.
- Per-row scale in `hooks.inject`, so items with different residual norms can be batched, with an
  equivalence test against per-item injection.

**Changed**
- Section 6 of `PREREG_readout_gap.md` now freezes a per-model band *rule* rather than an alpha
  number, because one alpha is an eightfold different intervention across models.
- `SELF_REPORT_PROBE` removed. It duplicated `SELF_REPORT_PROBES["state"]` and was not covered by
  `frozen_hash()`, so editing it would have shipped a changed probe under an unchanged hash.

**Findings**
- The preregistered headline was **refuted in direction**: forced-choice argmax over-reports a mass
  shift near a decision boundary rather than under-reporting it.
- The surviving effect is a **neutral floor**, localized to preference tuning, robust across seven
  injection depths, and present in the representation while absent from the option readout.

**Fixed**
- A headline check that printed "write the sentence" on a refuted claim.
- A capability gate that passed on `+0.0000` because the interval excluded zero while the point
  estimate did not clear any magnitude.
- A preregistered contrast with an inverted sign, disclosed in `RESULTS_shell.md` because
  correcting it changed a verdict in our favour.

## v0.4.1 (2026-08-01)

**Added**
- Addendum to `RESULTS_shell.md` testing the boundary-geometry objection to the SHELL result, on
  committed artifacts and labelled exploratory. The negative option's within-set log-odds do not
  rise sub-threshold under negative injection; they move slightly away (-0.086 against matched
  random, odds x0.918). The apparent rise in raw negative mass is a renormalization artifact of
  positive mass collapsing into neutral. This sharpens the dissociation to opposite signs.

**Changed**
- The correlated-feature caveat on SHELL is now stated as the weakest joint in the argument rather
  than a footnote: orthogonality removes the injected vector, not directions correlated with it, so
  "carries the state" is not yet separable from "carries the wake of the push". The experiment that
  would separate them is named in `PLAN.md` day 1b.
- `PLAN.md` records that directions have always been fit per model and never transferred, so
  Llama's inertness is not a porting failure.

## v0.1.0 (2026-07-31)

Instrument construction before the confirmatory runs: frozen preregistration, stimuli, injection hooks,
judge-free scorers, and the unit and negative tests. No experiment run.
