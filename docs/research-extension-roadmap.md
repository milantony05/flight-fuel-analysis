# Model reliability and publication roadmap

## Proposed research question

Can coverage-constrained selective regression preserve access across aircraft
and interval-duration groups while controlling prediction risk under missing
ADS-B observations and temporal or route distribution shift?

## Completed engineering milestone

- Integrated the original analysis with the teammate's six extension commits.
- Added a mixed-integer minimum-risk selector with explicit overall, duration
  and sufficiently represented aircraft-group constraints.
- Added decision explanations that distinguish low-risk inclusion from
  constraint-driven inclusion.
- Added five-fold out-of-fold isotonic error calibration and exploratory
  scaled-ratio 90% interval diagnostics without fitting a row's uncertainty on its own label.
- Added row-bootstrap confidence intervals, calibration bins, subgroup
  reliability tables and an additional calibrated group-floor policy.
- Added a self-contained professional dashboard with four policy choices,
  seven coverage targets, aircraft and duration filters, historical streaming
  replay, per-row uncertainty and an accuracy–coverage frontier.
- Added a generated result report, Python packaging, a one-command workflow
  and continuous integration.
- Added cross-fitted duration- and aircraft-Mondrian interval comparisons,
  aircraft-held-out risk-calibrator transfer diagnostics, natural ADS-B
  availability analysis and deterministic risk-score perturbation tests.
- Added a prospective confirmatory protocol, external-batch schema validation,
  stable dataset fingerprints, a provider-safe streaming contract and a
  LaTeX manuscript draft.

## Next scientific milestone: independently validated uncertainty

1. Freeze a chronological calibration block that is not used for fitting the
   fuel model or the risk model.
2. Replace the current cross-fitted diagnostic with a strictly earlier split-
   conformal calibration block and lock its quantile before final evaluation.
3. Report empirical coverage and interval width overall and by aircraft,
   duration, route familiarity and ADS-B availability.
4. Compare global, Mondrian/group-conditional and risk-adaptive conformal
   variants. Clearly report small-group instability.
5. Never calibrate intervals on the committed September validation labels used
   for the headline selector comparison.

The planned hypotheses and settings are recorded in
`configs/confirmatory-protocol.json`. They remain pending because no post-freeze
confirmatory labels are present in this repository.

## Next generalization milestone

- Register the protocol before opening the final labels.
- Evaluate time-disjoint, route-disjoint and—where identifiers permit—airframe-
  disjoint splits.
- Add stronger tabular baselines and ablations for type, duration, trajectory
  summaries and missingness indicators.
- Repeat selection after every model refit; do not reuse fixed masks when making
  generalization claims.
- Report flight-cluster uncertainty and multiple-comparison-aware subgroup
  analysis.

## Real-time architecture

The current dataset supplies completed ACARS-defined interval boundaries. A
real-time adapter therefore cannot reuse the retrospective inference contract
unchanged.

Build two adapters behind one interface:

1. **Historical replay** — the default reproducible mode. Replay cached, pre-scored historical intervals, reveal their committed labels after the simulated interval closes,
   and compute delayed evaluation metrics.
2. **Approved live telemetry** — an optional adapter enabled only after the
   provider's authentication, rate-limit and licensing requirements are met.
   Show predictions, uncertainty and abstention decisions, but never invent a
   real-time reference fuel value.

Both adapters should feed the same components:

```text
state source -> flight state store -> rolling feature extractor
             -> versioned model bundle -> uncertainty + drift checks
             -> constrained decision service -> WebSocket dashboard
```

## Publication boundary

The selector, exploratory calibration diagnostics and dashboard are meaningful engineering
and experimental contributions, not a guarantee of journal acceptance. A
strong paper requires a
clear comparison with selective-regression and fairness-constrained baselines,
an untouched confirmatory dataset, calibrated uncertainty, independent
reproduction and appropriately narrow aviation claims.

The current cross-fitted calibrator is learned from the September evaluation
batch and its scaled-ratio quantiles do not establish a formal conformal
coverage guarantee. Its policy comparison can have indirect batch-level label
dependence. See the evaluation boundary in the README.
