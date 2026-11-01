# Publication evidence plan

## Defensible paper position

**Coverage-constrained selective regression for aircraft fuel estimation under
ADS-B missingness and subgroup shift.** The contribution is the joint audit of
prediction error, abstention, represented fuel, subgroup access and empirical
uncertainty validity. It is not a claim that the underlying fuel estimator is
the best available model.

The current evidence is exploratory because the policy and uncertainty variants
were developed after September and October 2025 results had been inspected. The
frozen protocol in `configs/confirmatory-protocol.json` defines the hypotheses
and primary settings that must be applied unchanged to new labels.

## Status of the eight publication milestones

| Milestone | Implemented now | Evidence still required |
| --- | --- | --- |
| 1. Untouched evaluation | Schema validator, stable dataset fingerprint and frozen hypotheses | A post-freeze labeled period or independent operator/geography |
| 2. Generalization | Aircraft-held-out **risk-calibrator** transfer diagnostic; existing seen/unseen-route descriptive audit | Refit the complete fuel model under route-, time- and aircraft-disjoint splits |
| 3. Strong/physics models | Locked comparison list and common evaluation contract | Raw trajectories, weather and OpenAP inputs to fit CatBoost, temporal and physics-guided hybrids |
| 4. Calibrated uncertainty | Global, duration-Mondrian and aircraft-Mondrian cross-fitted diagnostics | Strict earlier split-conformal calibration followed by untouched confirmation; shift-weighted sensitivity |
| 5. Missingness and shift | Natural ADS-B availability audit and score-noise sensitivity | Contiguous raw-trajectory outage simulation, receiver-region shift and drift tests |
| 6. Fair selective policy | Risk, greedy floor, mixed-integer optimum and calibrated floor across seven coverage levels | Confirm the locked policy on new labels and compare with additional fair-selective baselines |
| 7. Statistical rigor | Deterministic seeds, row-bootstrap uncertainty, leakage tests and prospective plan | Flight-cluster 2,000-repetition bootstrap, repeated refits, Holm correction and effect sizes on confirmation data |
| 8. Real-time engineering | Provider-safe event contract, delayed-label semantics and historical streaming replay | Written provider permission, credentials, rate-limit implementation, model registry and delayed ground-truth feed |

## New exploratory findings

- Duration-Mondrian intervals reach 90.23% coverage with 422.54 kg mean width,
  compared with 89.95% and 449.97 kg for the global construction.
- Aircraft-Mondrian intervals reach 90.12% pooled coverage but are wider
  (579.05 kg) and fall to 87.88% for the worst eligible type.
- Leave-one-aircraft-out risk calibration reaches 90.45% pooled coverage but
  only 75.76% for B744. The upstream fuel model was not held out.
- Zero-position intervals have 163.22 kg MAE and 84.88% interval coverage,
  compared with 68.84 kg and 92.06% for intervals with at least 51 positions.
- At 50% score-noise standard deviation, calibrated-floor selection retains
  76.80% mean Jaccard overlap and accepted MAE rises to 63.44 kg.

These are useful hypotheses and failure modes, not independent confirmation.

## Required external data contract

The confirmation table must contain unique interval and flight identifiers,
UTC boundaries, aircraft type, frozen-model predictions, reference fuel,
predicted error risk, ADS-B position count, route, and model version. Validate
it with `flight_fuel.research_protocol` before examining aggregate outcomes.
Record the returned table fingerprint in the experiment log.

## Live-data rule

The live adapter may display prediction, uncertainty and abstention immediately.
It must show reference fuel as unavailable until an authorized delayed label is
received. Historical replay can reveal the committed reference after simulated
interval closure. The two modes must never be visually or analytically mixed.


## September calibration caveat

The new calibrator uses September evaluation labels in other folds. A row's own
label is excluded from its isotonic fit, but it can change the scores of
competing rows in the batch and thus the calibrated policy's selections. Ratio
quantiles use scales fitted on the same training folds. Treat its accepted MAE
and 90% empirical interval coverage as exploratory diagnostics, not an
independent policy validation or a formal conformal guarantee. Fit the risk
calibrator and quantiles on earlier disjoint flights for confirmation.
