# Flight Fuel Analysis: Coverage and Error Audit

An interactive data science project that estimates fuel use for **completed flight intervals** from public flight tracking data, then checks **which flights those estimates cover and how accurate they are**. [Open the presentation demo](demo/index.html) or [walk through the results in Jupyter](notebooks/eligibility-walkthrough.ipynb).

## Optimization and reliability dashboard

The research extension adds a **coverage-constrained selector**, five-fold
out-of-fold isotonic risk calibration and empirical scaled prediction intervals.
The visual demo embeds its aircraft image and lets you inspect historical
intervals alongside full-period prediction and coverage results. It remains fully
self-contained in [`demo/index.html`](demo/index.html).

The exploratory research layer additionally compares global, duration-Mondrian
and aircraft-Mondrian uncertainty, evaluates risk-calibrator transfer when each
sufficiently large aircraft type is held out, audits natural ADS-B availability,
and stress-tests the selector under score perturbations. A frozen prospective
protocol in [`configs/confirmatory-protocol.json`](configs/confirmatory-protocol.json)
defines the hypotheses and evidence required for a later untouched batch.

**Evaluation boundary:** The original fuel and risk models use earlier time
blocks, but this extension fits its uncertainty calibrators on other folds of
the September evaluation rows. Even though each row's own interval fit excludes
its label, the calibrated selection policy is learned from the evaluation
batch as a whole: a row's label can influence the competing rows' scores.
Its accepted-error comparison is therefore exploratory, not an independent
validation of a deployable calibrated policy. The ratio quantiles also use
in-sample fitted scales, so the reported 90% coverage is empirical rather than
a finite-sample conformal guarantee. A future test must train and calibrate on
earlier disjoint flights and freeze decisions before inspecting new labels.

The replay uses committed historical benchmark rows; it is not a live aircraft
feed. A live system would require rolling-window retraining, approved telemetry
access, calibrated uncertainty and a new independently frozen evaluation period.
See [`UPSTREAM_ATTRIBUTION.md`](UPSTREAM_ATTRIBUTION.md) for the boundary between
the upstream project and this fork's additions.

### New robustness findings

- Global cross-fitted intervals reach **89.95%** empirical coverage with a
  **449.97 kg** mean width.
- Duration-Mondrian intervals reach **90.23%** coverage with a narrower
  **422.54 kg** mean width and only a **0.26 percentage-point** eligible-group
  coverage gap on this exploratory benchmark.
- Aircraft-Mondrian intervals are wider (**579.05 kg**) and still reach only
  **87.88%** coverage for the worst eligible aircraft group.
- When the risk calibrator is trained without each tested aircraft type in
  turn, pooled coverage is **90.45%**, but B744 coverage falls to **75.76%**.
  The fuel estimator itself is not held out in this diagnostic.
- Zero-position intervals have **163.22 kg** MAE versus **68.84 kg** for
  intervals with at least 51 ADS-B positions. Natural missingness is not random,
  so this is an association rather than a causal outage effect.

## The problem

Public ADS-B flight tracking data contain aircraft positions, but no measured fuel consumption. Researchers can estimate fuel use with a model; if they keep only its easiest predictions, the reported error looks better while longer intervals and some aircraft types largely disappear from the sample. That can distort conclusions drawn from the selected data.

**Research question:** For a batch of completed flight intervals, how does restricting estimates to lower-risk intervals affect prediction error, coverage across aircraft and interval lengths, and the share of the sample's observed fuel represented?

**Who can use this:** aviation researchers and public flight-data curators evaluating whether an estimated-fuel dataset adequately covers the flights they intend to study. The project also demonstrates to data science reviewers how model selection, missing data, subgroup analysis and honest evaluation fit into a reproducible workflow. It is not an airline fuel-planning tool.

## What I built

1. Verify the public dataset downloads and extract ADS-B summaries within each supplied flight interval. The reference fuel value comes from the difference between fuel-on-board reports; it is an imperfect label, not a fuel-flow sensor reading.
2. Train a fuel estimator on April–August 2025 data using aircraft type, interval duration and ADS-B summaries. Use a training-only aircraft-type median-rate fallback when an interval has no usable ADS-B positions.
3. Train a separate error-ranking model on early-September residuals. For a batch, rank intervals by predicted error and select an overall percentage. An optional **group floor** ensures a minimum accepted fraction in each duration group and aircraft type with at least 50 intervals in that batch.
4. Compare prediction error and access on later labeled intervals. The browser dashboard recalculates the selection locally as you change the overall target or group floor; labels are used to *evaluate* its choices, never to choose intervals.

The dashboard contains pre-scored benchmark intervals and works without a server. It does not accept new raw flights or run live inference.

## Main finding

At the same **75.01%** overall acceptance on **7,619 September validation intervals**, giving each eligible group at least 50% access changes the sample substantially:

| Measure | Lowest predicted error only | With 50% group floor |
| --- | ---: | ---: |
| Mean absolute error on accepted intervals | 44.23 kg | 54.02 kg |
| 15–60-minute intervals accepted | 0.60% | 50.12% |
| Boeing 747-400 (B744) intervals accepted | 1 of 66 | 33 of 66 |
| Share of this sample's labeled fuel represented | 29.77% | 41.51% |

**Interpretation:** broader access costs accuracy. The 33 accepted B744 intervals under the floor have **609.90 kg** mean absolute error. This is an important warning, not a success claim for those estimates. On *all* September validation intervals, the fuel model's MAE is **96.07 kg**, RMSE **281.69 kg** and R² **0.872**; its 95th percentile absolute error is **372.75 kg**. R² and selective MAE alone conceal large individual and subgroup errors. [Full results and residual breakdowns](reports/eligibility-results.json) are committed for inspection.

## Explore and reproduce

- **Use one demo:** download and open [`demo/index.html`](demo/index.html) in Chrome, Edge or Firefox. It works offline; no Python installation, server or API key is needed. GitHub's file viewer displays source, so download the file before opening it.
- **Three-minute presentation:** click through the five project stages, then choose **Full benchmark → Prioritize accuracy → Broaden representation → Inspect the B744**. Hover or click the scatter plot to inspect an interval. Change the retention target to update the full-batch comparison. The cohort cards follow your aircraft, duration and tracking filters; the policy comparison always uses the full batch.
- **Presentation design:** one offline dashboard with an embedded aircraft hero, clickable project lifecycle, logarithmic prediction scatter plot, interval inspector, group-coverage comparison and explicit evaluation limits. All scripts, styles, imagery and benchmark data are embedded; no external fonts or chart CDNs are required.
- The controls filter committed historical predictions; they do not synthesize a new flight or estimate fuel from user-entered payload or weather. The dataset and trained model do not support those inputs.
- `src/flight_fuel/demo-template.html` is the maintained source for this one-file demo. `data/benchmark-source.html` is the committed benchmark source required by analysis and replay tests.
- Open the [Jupyter walkthrough](notebooks/eligibility-walkthrough.ipynb) for the result tables, access chart, residual diagnostics and route-overlap check.
- To rebuild the analysis from the publisher's data, use Python 3.11+ and allow roughly 5 GB for the trajectory archives plus extraction space:

```bash
python -m pip install -r requirements.txt
python -m src.flight_fuel.data_audit --download
python -m src.flight_fuel.download_archives
python -m src.flight_fuel.features --splits train rank final
python -m src.flight_fuel.eligibility
python -m unittest discover -s tests
```

To regenerate the reliability workspace without downloading the raw trajectory archives:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m flight_fuel.advanced
.venv/bin/python -m http.server 8000
```

Then open `http://localhost:8000/demo/index.html`, or open the generated file
directly because it is self-contained. The command reads the committed benchmark
payload and regenerates both `reports/advanced-results.json` and
`demo/index.html`.

To inspect the provider-safe event contract without revealing historical
labels, run:

```bash
.venv/bin/python -m flight_fuel.streaming --limit 20 --withhold-labels
```

The live-provider class intentionally refuses to operate without an explicitly
approved provider. It never invents a real-time reference-fuel value.

Raw data and extracted features are excluded from Git. Downloads are checked against publisher MD5 values. The final analysis command regenerates `reports/eligibility-results.json` and `data/benchmark-source.html`.

## Scope and limitations

- The policy was developed **after earlier October results were examined**. Its September and October comparisons are exploratory; October is not a fresh independent confirmation. The revised model also disables a row-random internal early-stopping split used in historical Phase 4, so the historical score is not a controlled policy-only comparison.
- Aircraft routes overlap across periods. In the exploratory October batch, MAE is **81.15 kg** on intervals from routes seen in training and **174.77 kg** on unseen routes. This comparison is affected by aircraft and duration mix, but warns against assuming transfer.
- Interval boundaries come from the completed fuel-report dataset. Payload, weight, wind and weather are absent, and raw altitude/speed units are unverified. The labels are fuel-on-board differences; neither per-flight operational fuel needs nor emissions inventories are validated here.
- The labeled-fuel share is known only for this evaluation sample. The error score ranks cases; it is not a calibrated confidence interval or an individual accuracy guarantee. The dataset contains many intervals shorter than the publisher README's stated five-minute lower bound.

The contribution is a **dataset-specific, transparent coverage/error audit**, not a novel fuel-prediction algorithm. See the [methodology](docs/methods.md), [related work](docs/literature.md) and [self-critical audit](docs/pre-submission-audit.md) for design choices, checks and outstanding research work.

## Project structure and data source

| Path | Purpose |
| --- | --- |
| [`src/flight_fuel/`](src/flight_fuel/) | Download checks, feature extraction, baselines, evaluation and batch selection. |
| [`demo/`](demo/) | Self-contained interactive dashboard and its maintained template. |
| [`notebooks/`](notebooks/) | Presentation-friendly analysis of committed results. |
| [`reports/`](reports/) | Generated result JSON and the historical Phase 4 record. |
| [`tests/`](tests/) | Decision-rule, accounting and dashboard checks. |
| [`paper/`](paper/) | Exploratory LaTeX manuscript draft with explicit validation limits. |

Data: [EUROCONTROL PRC 2025 Aircraft Fuel Burn Estimation Data Challenge](https://doi.org/10.5281/zenodo.19184662), combining ACARS-derived fuel labels with OpenSky ADS-B records and metadata, licensed CC BY 4.0. Cite the [dataset paper](https://doi.org/10.59490/joas.2026.8750). This repository does not contain the raw dataset.
