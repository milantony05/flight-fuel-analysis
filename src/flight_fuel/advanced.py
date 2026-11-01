"""Coverage-constrained analysis and professional dashboard generator.

This module operates on the committed, pre-scored September benchmark rows.
It adds exact mixed-integer selection, cross-fitted uncertainty calibration,
conformal-style intervals, decision explanations and an offline streaming
replay. Replayed records remain historical and labels never enter a row's own
uncertainty fit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression

from .eligibility import DURATION_LABELS, describe, duration_groups, select

ROW_COLUMNS = (
    "aircraft_type",
    "duration_min",
    "risk_score_kg",
    "predicted_fuel_kg",
    "reference_fuel_kg",
    "adsb_position_count",
    "stable_tie",
)

COVERAGE_TARGETS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)


def load_demo_payload(path: Path) -> dict:
    """Read the JSON payload embedded in the generated offline dashboard."""
    source = path.read_text()
    marker = "const DATA="
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"No embedded DATA payload in {path}")
    payload, _ = json.JSONDecoder().raw_decode(source[start + len(marker):])
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError("Dashboard payload must contain rows")
    return payload


def rows_frame(payload: dict) -> pd.DataFrame:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dashboard payload rows must be a nonempty list")
    frame = pd.DataFrame(rows, columns=ROW_COLUMNS)
    numeric = [name for name in ROW_COLUMNS if name != "aircraft_type"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Advanced analysis requires finite committed demo rows")
    frame["flight_id"] = [f"replay-{i:05d}" for i in range(len(frame))]
    frame["fuel_kg"] = frame.reference_fuel_kg.astype(float)
    return frame


def cross_fitted_uncertainty(
    risk: np.ndarray,
    predicted: np.ndarray,
    truth: np.ndarray,
    ties: np.ndarray,
    *,
    folds: int = 5,
    interval_level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Explore risk and interval width without fitting on a row's own label.

    Each deterministic fold receives predictions from an isotonic model fitted
    on the other folds. A scaled conformal quantile from those training folds
    then produces an out-of-fold interval half-width for the held-out rows.
    The ratio scores use in-sample fitted scales; coverage is empirical and
    no finite-sample conformal guarantee is asserted.
    """
    risk, predicted, truth, ties = (
        np.asarray(values, dtype=float) for values in (risk, predicted, truth, ties)
    )
    n = len(risk)
    if n < folds or folds < 2 or any(len(values) != n for values in (predicted, truth, ties)):
        raise ValueError("Cross-fitted uncertainty requires equally sized arrays and at least two folds")
    if not 0 < interval_level < 1:
        raise ValueError("interval_level must be between zero and one")
    absolute_error = np.abs(predicted - truth)
    fold_id = np.minimum((np.mod(ties, 1.0) * folds).astype(int), folds - 1)
    calibrated = np.empty(n, dtype=float)
    half_width = np.empty(n, dtype=float)
    for fold in range(folds):
        held_out = fold_id == fold
        train = ~held_out
        if not held_out.any() or train.sum() < 2:
            raise ValueError("Stable tie values do not support the requested folds")
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        model.fit(risk[train], absolute_error[train])
        calibrated[held_out] = model.predict(risk[held_out])
        train_scale = np.maximum(model.predict(risk[train]), 1.0)
        ratios = absolute_error[train] / train_scale
        quantile = min(1.0, math.ceil((len(ratios) + 1) * interval_level) / len(ratios))
        multiplier = float(np.quantile(ratios, quantile, method="higher"))
        half_width[held_out] = np.maximum(calibrated[held_out], 1.0) * multiplier

    covered = absolute_error <= half_width
    edges = np.quantile(calibrated, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    bins = []
    if len(edges) > 1:
        bin_id = np.clip(np.digitize(calibrated, edges[1:-1], right=True), 0, len(edges) - 2)
        for index in range(len(edges) - 1):
            member = bin_id == index
            if member.any():
                bins.append({
                    "n": int(member.sum()),
                    "predicted_error_kg": float(calibrated[member].mean()),
                    "observed_mae_kg": float(absolute_error[member].mean()),
                    "interval_coverage": float(covered[member].mean()),
                })
    expected_calibration_error = float(sum(
        (row["n"] / n) * abs(row["predicted_error_kg"] - row["observed_mae_kg"])
        for row in bins
    ))
    diagnostics = {
        "method": "5-fold out-of-fold isotonic calibration with scaled conformal intervals",
        "folds": folds,
        "nominal_interval_coverage": interval_level,
        "observed_interval_coverage": float(covered.mean()),
        "mean_interval_width_kg": float(2 * half_width.mean()),
        "median_interval_width_kg": float(2 * np.median(half_width)),
        "expected_calibration_error_kg": expected_calibration_error,
        "bins": bins,
    }
    return calibrated, half_width, diagnostics


def _finite_sample_quantile(values: np.ndarray, level: float) -> float:
    """Return the conservative split-conformal quantile."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 < level < 1:
        raise ValueError("Conformal scores and level must be valid")
    quantile = min(1.0, math.ceil((len(values) + 1) * level) / len(values))
    return float(np.quantile(values, quantile, method="higher"))


def cross_fitted_mondrian_uncertainty(
    risk: np.ndarray,
    predicted: np.ndarray,
    truth: np.ndarray,
    ties: np.ndarray,
    groups: np.ndarray,
    *,
    group_label: str,
    folds: int = 5,
    interval_level: float = 0.9,
    min_group_n: int = 50,
) -> tuple[np.ndarray, dict]:
    """Cross-fit group-conditional interval scales with an honest fallback.

    The risk-to-error mapping is fitted without the held-out fold. Within each
    training fold, a group-specific conformal multiplier is used only when the
    group has enough calibration rows; otherwise the global multiplier is used.
    This is an exploratory Mondrian diagnostic, not a future-period guarantee.
    """
    risk, predicted, truth, ties = (
        np.asarray(values, dtype=float) for values in (risk, predicted, truth, ties)
    )
    groups = np.asarray(groups).astype(str)
    n = len(risk)
    if folds < 2 or n < folds or any(len(values) != n for values in (predicted, truth, ties, groups)):
        raise ValueError("Mondrian uncertainty inputs must have equal length")
    if min_group_n < 2:
        raise ValueError("min_group_n must be at least two")
    absolute_error = np.abs(predicted - truth)
    fold_id = np.minimum((np.mod(ties, 1.0) * folds).astype(int), folds - 1)
    half_width = np.empty(n, dtype=float)
    used_fallback = np.zeros(n, dtype=bool)
    for fold in range(folds):
        held_out = fold_id == fold
        train = ~held_out
        if not held_out.any() or train.sum() < 2:
            raise ValueError("Stable tie values do not support the requested folds")
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        model.fit(risk[train], absolute_error[train])
        train_scale = np.maximum(model.predict(risk[train]), 1.0)
        held_scale = np.maximum(model.predict(risk[held_out]), 1.0)
        train_ratios = absolute_error[train] / train_scale
        global_multiplier = _finite_sample_quantile(train_ratios, interval_level)
        held_indices = np.flatnonzero(held_out)
        for name in np.unique(groups[held_out]):
            local_train = train & (groups == name)
            local_held = groups[held_indices] == name
            target_indices = held_indices[local_held]
            if local_train.sum() >= min_group_n:
                multiplier = _finite_sample_quantile(
                    absolute_error[local_train]
                    / np.maximum(model.predict(risk[local_train]), 1.0),
                    interval_level,
                )
            else:
                multiplier = global_multiplier
                used_fallback[target_indices] = True
            half_width[target_indices] = held_scale[local_held] * multiplier

    covered = absolute_error <= half_width
    group_rows = []
    for name in sorted(np.unique(groups)):
        member = groups == name
        group_rows.append({
            "group": name,
            "n": int(member.sum()),
            "eligible_for_group_summary": bool(member.sum() >= min_group_n),
            "coverage": float(covered[member].mean()),
            "mean_interval_width_kg": float(2 * half_width[member].mean()),
            "fallback_fraction": float(used_fallback[member].mean()),
        })
    eligible_rows = [row for row in group_rows if row["eligible_for_group_summary"]]
    if not eligible_rows:
        eligible_rows = group_rows
    diagnostics = {
        "method": f"{folds}-fold cross-fitted {group_label} Mondrian scaled conformal",
        "nominal_interval_coverage": interval_level,
        "observed_interval_coverage": float(covered.mean()),
        "mean_interval_width_kg": float(2 * half_width.mean()),
        "median_interval_width_kg": float(2 * np.median(half_width)),
        "worst_group_coverage": float(min(row["coverage"] for row in eligible_rows)),
        "group_coverage_gap": float(
            max(row["coverage"] for row in eligible_rows)
            - min(row["coverage"] for row in eligible_rows)
        ),
        "fallback_fraction": float(used_fallback.mean()),
        "groups": group_rows,
    }
    return half_width, diagnostics


def leave_aircraft_out_risk_calibration(
    risk: np.ndarray,
    predicted: np.ndarray,
    truth: np.ndarray,
    aircraft: np.ndarray,
    *,
    interval_level: float = 0.9,
    min_type_n: int = 50,
) -> dict:
    """Test risk calibration on each sufficiently large unseen aircraft group.

    Only the uncertainty calibrator is held out here. The upstream fuel model
    may have seen the aircraft type, so this must not be presented as complete
    unseen-aircraft fuel-model validation.
    """
    risk, predicted, truth = (
        np.asarray(values, dtype=float) for values in (risk, predicted, truth)
    )
    aircraft = np.asarray(aircraft).astype(str)
    error = np.abs(predicted - truth)
    rows = []
    covered_parts = []
    width_parts = []
    for name, count in zip(*np.unique(aircraft, return_counts=True)):
        if count < min_type_n:
            continue
        held = aircraft == name
        train = ~held
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        model.fit(risk[train], error[train])
        train_scale = np.maximum(model.predict(risk[train]), 1.0)
        multiplier = _finite_sample_quantile(error[train] / train_scale, interval_level)
        half = np.maximum(model.predict(risk[held]), 1.0) * multiplier
        covered = error[held] <= half
        covered_parts.append(covered)
        width_parts.append(2 * half)
        rows.append({
            "aircraft_type": name,
            "n": int(count),
            "mae_kg": float(error[held].mean()),
            "interval_coverage": float(covered.mean()),
            "mean_interval_width_kg": float(2 * half.mean()),
        })
    if not rows:
        raise ValueError("No aircraft groups meet the holdout threshold")
    return {
        "scope": "Risk calibrator held out by aircraft type; fuel model not held out",
        "eligible_aircraft_types": len(rows),
        "rows_evaluated": int(sum(row["n"] for row in rows)),
        "pooled_interval_coverage": float(np.concatenate(covered_parts).mean()),
        "pooled_mean_interval_width_kg": float(np.concatenate(width_parts).mean()),
        "worst_aircraft_coverage": float(min(row["interval_coverage"] for row in rows)),
        "aircraft": sorted(rows, key=lambda row: row["interval_coverage"]),
    }


def _regression_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    residual = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    absolute = np.abs(residual)
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "mae_kg": float(absolute.mean()),
        "rmse_kg": float(np.sqrt(np.mean(residual ** 2))),
        "bias_kg": float(residual.mean()),
        "median_absolute_error_kg": float(np.median(absolute)),
        "p90_absolute_error_kg": float(np.quantile(absolute, 0.9)),
        "p95_absolute_error_kg": float(np.quantile(absolute, 0.95)),
        "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator else None,
    }


def _bootstrap_mae_ci(
    predicted: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    repetitions: int = 500,
    seed: int = 20251022,
) -> list[float]:
    error = np.abs(np.asarray(predicted)[mask] - np.asarray(truth)[mask])
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = error[rng.integers(0, len(error), len(error))].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _group_reliability(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    raw_risk: np.ndarray,
    calibrated_risk: np.ndarray,
    half_width: np.ndarray,
    min_type_n: int,
) -> dict:
    truth = frame.fuel_kg.to_numpy(dtype=float)
    error = predicted - truth
    covered = np.abs(error) <= half_width

    def summarize(name: str, member: np.ndarray) -> dict:
        return {
            "group": name,
            "n": int(member.sum()),
            "mae_kg": float(np.abs(error[member]).mean()),
            "rmse_kg": float(np.sqrt(np.mean(error[member] ** 2))),
            "bias_kg": float(error[member].mean()),
            "mean_raw_risk_kg": float(raw_risk[member].mean()),
            "mean_calibrated_risk_kg": float(calibrated_risk[member].mean()),
            "interval_coverage": float(covered[member].mean()),
        }

    duration = duration_groups(frame.duration_min.to_numpy(dtype=float))
    duration_rows = [summarize(name, duration == name) for name in DURATION_LABELS if np.any(duration == name)]
    types = frame.aircraft_type.astype(str).to_numpy()
    names, counts = np.unique(types, return_counts=True)
    aircraft_rows = [
        summarize(name, types == name)
        for name, count in zip(names, counts)
        if count >= min_type_n
    ]
    aircraft_rows.sort(key=lambda row: (-row["n"], row["group"]))
    return {"duration": duration_rows, "aircraft": aircraft_rows}


def _missingness_reliability(
    frame: pd.DataFrame,
    predicted: np.ndarray,
    calibrated_risk: np.ndarray,
    half_width: np.ndarray,
) -> list[dict]:
    """Summarize natural ADS-B availability without treating it as random."""
    positions = frame.adsb_position_count.to_numpy(dtype=int)
    names = np.select(
        [positions == 0, positions <= 10, positions <= 50],
        ["Zero positions", "Sparse (1–10)", "Partial (11–50)"],
        default="Dense (51+)",
    )
    truth = frame.fuel_kg.to_numpy(dtype=float)
    error = np.asarray(predicted, dtype=float) - truth
    covered = np.abs(error) <= half_width
    rows = []
    for name in ("Zero positions", "Sparse (1–10)", "Partial (11–50)", "Dense (51+)"):
        member = names == name
        if not member.any():
            continue
        rows.append({
            "group": name,
            "n": int(member.sum()),
            "mae_kg": float(np.abs(error[member]).mean()),
            "rmse_kg": float(np.sqrt(np.mean(error[member] ** 2))),
            "mean_calibrated_risk_kg": float(calibrated_risk[member].mean()),
            "interval_coverage": float(covered[member].mean()),
            "mean_interval_width_kg": float(2 * half_width[member].mean()),
        })
    return rows


def _risk_perturbation_sensitivity(
    frame: pd.DataFrame,
    calibrated_risk: np.ndarray,
    predicted: np.ndarray,
    ties: np.ndarray,
    *,
    target: float,
    floor: float,
    min_type_n: int,
    repetitions: int = 100,
    seed: int = 20251022,
) -> dict:
    """Stress-test selection stability under deterministic score noise."""
    types = frame.aircraft_type.to_numpy(dtype=str)
    duration = frame.duration_min.to_numpy(dtype=float)
    truth = frame.fuel_kg.to_numpy(dtype=float)
    baseline, _ = select(
        calibrated_risk, types, duration, ties, target, floor, min_type_n
    )
    groups = duration_groups(duration)
    score_scale = max(float(np.std(calibrated_risk)), 1.0)
    rng = np.random.default_rng(seed)
    scenarios = []
    for fraction in (0.10, 0.25, 0.50):
        jaccard, mae, gap = [], [], []
        for _ in range(repetitions):
            perturbed = np.maximum(
                0.0,
                calibrated_risk
                + rng.normal(0.0, fraction * score_scale, len(calibrated_risk)),
            )
            mask, _ = select(perturbed, types, duration, ties, target, floor, min_type_n)
            union = np.logical_or(mask, baseline).sum()
            jaccard.append(float(np.logical_and(mask, baseline).sum() / union))
            mae.append(float(np.abs(predicted[mask] - truth[mask]).mean()))
            coverages = [
                float(mask[groups == name].mean())
                for name in DURATION_LABELS
                if np.any(groups == name)
            ]
            gap.append(max(coverages) - min(coverages))
        scenarios.append({
            "score_noise_fraction_of_sd": fraction,
            "repetitions": repetitions,
            "mean_selection_jaccard": float(np.mean(jaccard)),
            "selection_jaccard_95pct_range": [
                float(value) for value in np.quantile(jaccard, [0.025, 0.975])
            ],
            "mean_accepted_mae_kg": float(np.mean(mae)),
            "accepted_mae_95pct_range_kg": [
                float(value) for value in np.quantile(mae, [0.025, 0.975])
            ],
            "mean_duration_coverage_gap": float(np.mean(gap)),
        })
    return {
        "scope": "Exploratory sensitivity; perturbation is not a physical ADS-B outage model",
        "baseline_selected": int(baseline.sum()),
        "scenarios": scenarios,
    }


def _constraint_members(
    types: np.ndarray,
    duration: np.ndarray,
    min_type_n: int,
) -> list[tuple[str, np.ndarray]]:
    groups = duration_groups(duration)
    constraints = [
        (f"duration:{name}", groups == name)
        for name in DURATION_LABELS
        if np.any(groups == name)
    ]
    names, counts = np.unique(types.astype(str), return_counts=True)
    constraints.extend(
        (f"aircraft:{name}", types.astype(str) == name)
        for name, count in zip(names, counts)
        if count >= min_type_n
    )
    return constraints


def constrained_select(
    scores: np.ndarray,
    types: np.ndarray,
    duration: np.ndarray,
    ties: np.ndarray,
    target: float,
    floor: float,
    min_type_n: int = 50,
    *,
    time_limit_s: float = 60,
) -> tuple[np.ndarray, dict]:
    """Minimize total risk subject to overall and subgroup coverage constraints.

    A large per-row count penalty first minimizes the number of selected rows;
    predicted risk then breaks ties among equal-size feasible selections. This
    allows floors to exceed the nominal budget while avoiding unnecessary rows.
    """
    scores, types, duration, ties = (np.asarray(x) for x in (scores, types, duration, ties))
    n = len(scores)
    if not n or any(len(x) != n for x in (types, duration, ties)):
        raise ValueError("Nonempty, equally sized arrays required")
    if not np.isfinite(scores).all() or not np.isfinite(ties).all():
        raise ValueError("Scores and ties must be finite")
    if (ties < 0).any() or (ties >= 1).any():
        raise ValueError("Stable tie values must be in [0, 1)")
    if not (0 < target <= 1 and 0 <= floor <= 1 and min_type_n >= 1):
        raise ValueError("Invalid target, floor or minimum aircraft group size")

    span = float(np.ptp(scores))
    normalized = np.zeros(n) if span == 0 else (scores - float(scores.min())) / span
    count_penalty = float(n + 1)
    objective = count_penalty + normalized + ties.astype(float) * 1e-9

    members = _constraint_members(types, duration, min_type_n)
    rows = [np.ones(n, dtype=float)]
    lower = [math.ceil(target * n)]
    names = ["overall"]
    for name, mask in members:
        rows.append(mask.astype(float))
        lower.append(math.ceil(floor * int(mask.sum())))
        names.append(name)
    matrix = csr_matrix(np.vstack(rows))
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=np.uint8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(matrix, np.asarray(lower, dtype=float), np.full(len(rows), np.inf)),
        options={"time_limit": float(time_limit_s)},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Constrained selector failed: {result.message}")
    selected = result.x >= 0.5
    achieved = matrix @ selected.astype(float)
    details = {
        "method": "mixed_integer_minimum_risk",
        "solver": "scipy.optimize.milp (HiGHS)",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "target_count": int(lower[0]),
        "selected_count": int(selected.sum()),
        "floor_forced_over_budget": bool(selected.sum() > lower[0]),
        "objective_predicted_risk_sum_kg": float(scores[selected].sum()),
        "constraints": {
            name: {"required": int(required), "achieved": int(actual)}
            for name, required, actual in zip(names, lower, achieved)
        },
    }
    return selected, details


def decision_reasons(
    selected: np.ndarray,
    risk_only: np.ndarray,
    types: np.ndarray,
    duration: np.ndarray,
    min_type_n: int,
) -> list[str]:
    """Create honest, non-causal explanations for selection decisions."""
    selected, risk_only = np.asarray(selected, dtype=bool), np.asarray(risk_only, dtype=bool)
    types = np.asarray(types).astype(str)
    durations = duration_groups(np.asarray(duration, dtype=float))
    names, counts = np.unique(types, return_counts=True)
    protected = {name for name, count in zip(names, counts) if count >= min_type_n}
    reasons = []
    for keep, low_risk, aircraft, duration_name in zip(selected, risk_only, types, durations):
        if not keep:
            reasons.append("not selected at this coverage target")
        elif low_risk:
            reasons.append("selected by low predicted-error rank")
        else:
            memberships = [f"duration coverage ({duration_name})"]
            if aircraft in protected:
                memberships.append(f"aircraft coverage ({aircraft})")
            reasons.append("coverage-constrained optimum; " + " and ".join(memberships))
    return reasons


def _policy_summary(
    frame: pd.DataFrame,
    mask: np.ndarray,
    predicted: np.ndarray,
    risk: np.ndarray,
    policy: dict,
) -> dict:
    summary = describe(frame, predicted, risk, mask, policy)
    duration_coverages = [group["coverage"] for group in summary["groups"]["duration"].values()]
    protected = [
        group["coverage"]
        for group in summary["groups"]["aircraft_type"].values()
        if group["n"] >= int(policy.get("min_type_n", 50))
    ]
    summary["mean_predicted_risk_kg"] = float(np.mean(risk[mask]))
    summary["duration_coverage_gap"] = float(max(duration_coverages) - min(duration_coverages))
    summary["protected_aircraft_coverage_gap"] = (
        float(max(protected) - min(protected)) if protected else None
    )
    protected_rows = list(summary["groups"]["duration"].values()) + [
        group
        for group in summary["groups"]["aircraft_type"].values()
        if group["n"] >= int(policy.get("min_type_n", 50))
    ]
    accepted_group_errors = [
        group["accepted_mae_kg"]
        for group in protected_rows
        if group["accepted_mae_kg"] is not None
    ]
    summary["minimum_protected_group_coverage"] = float(
        min(group["coverage"] for group in protected_rows)
    )
    summary["worst_group_accepted_mae_kg"] = float(max(accepted_group_errors))
    summary["accepted_mae_95pct_row_bootstrap_ci_kg"] = _bootstrap_mae_ci(
        predicted, frame.fuel_kg.to_numpy(dtype=float), mask
    )
    return summary


def _selective_curve_summary(pareto: list[dict]) -> list[dict]:
    """Summarize the complete risk–coverage curve for every policy."""
    rows = []
    for name in sorted({row["policy"] for row in pareto}):
        values = sorted(
            (row for row in pareto if row["policy"] == name),
            key=lambda row: row["actual_coverage"],
        )
        coverage = np.asarray([row["actual_coverage"] for row in values], dtype=float)
        risk = np.asarray([row["accepted_mae_kg"] for row in values], dtype=float)
        span = float(coverage[-1] - coverage[0])
        area = float(np.sum(np.diff(coverage) * (risk[:-1] + risk[1:]) / 2))
        rows.append({
            "policy": name,
            "coverage_min": float(coverage[0]),
            "coverage_max": float(coverage[-1]),
            "risk_coverage_auc_kg": area,
            "mean_curve_mae_kg": area / span if span else float(risk[0]),
            "mae_at_full_coverage_kg": float(risk[-1]),
        })
    return rows


def analyze(
    payload: dict,
    *,
    target: float = 0.75,
    floor: float = 0.5,
    min_type_n: int = 50,
) -> tuple[dict, dict]:
    frame = rows_frame(payload)
    scores = frame.risk_score_kg.to_numpy(dtype=float)
    types = frame.aircraft_type.to_numpy(dtype=str)
    duration = frame.duration_min.to_numpy(dtype=float)
    ties = frame.stable_tie.to_numpy(dtype=float)
    predicted = frame.predicted_fuel_kg.to_numpy(dtype=float)
    truth = frame.fuel_kg.to_numpy(dtype=float)
    calibrated_risk, interval_half_width, uncertainty = cross_fitted_uncertainty(
        scores, predicted, truth, ties
    )
    duration_names = duration_groups(duration)
    _, duration_mondrian = cross_fitted_mondrian_uncertainty(
        scores, predicted, truth, ties, duration_names,
        group_label="duration", min_group_n=min_type_n,
    )
    _, aircraft_mondrian = cross_fitted_mondrian_uncertainty(
        scores, predicted, truth, ties, types,
        group_label="aircraft", min_group_n=min_type_n,
    )

    targets = tuple(sorted(set(COVERAGE_TARGETS + (target,))))
    selection_grid: dict[str, dict[str, str]] = {}
    policy_grid: dict[str, dict[str, dict]] = {}
    pareto = []
    policies = None
    default_masks = None
    compact_fields = (
        "accepted", "coverage", "accepted_mae_kg", "rejected_mae_kg",
        "accepted_reference_fuel_fraction", "mean_predicted_risk_kg",
        "duration_coverage_gap", "protected_aircraft_coverage_gap",
        "minimum_protected_group_coverage", "worst_group_accepted_mae_kg",
        "accepted_mae_95pct_row_bootstrap_ci_kg",
    )
    for coverage in targets:
        risk_mask, risk_details = select(scores, types, duration, ties, coverage, 0, min_type_n)
        greedy_mask, greedy_details = select(scores, types, duration, ties, coverage, floor, min_type_n)
        optimized_mask, optimized_details = constrained_select(
            scores, types, duration, ties, coverage, floor, min_type_n
        )
        calibrated_mask, calibrated_details = select(
            calibrated_risk, types, duration, ties, coverage, floor, min_type_n
        )
        masks = {
            "risk": risk_mask,
            "greedy": greedy_mask,
            "optimized": optimized_mask,
            "calibrated": calibrated_mask,
        }
        summaries = {
            "risk_only": _policy_summary(
                frame, risk_mask, predicted, scores,
                {"method": "risk_rank", "floor": 0, "min_type_n": min_type_n, **risk_details},
            ),
            "greedy_floor": _policy_summary(
                frame, greedy_mask, predicted, scores,
                {"method": "duration_then_aircraft_greedy", "floor": floor,
                 "min_type_n": min_type_n, **greedy_details},
            ),
            "optimized_floor": _policy_summary(
                frame, optimized_mask, predicted, scores,
                {"floor": floor, "min_type_n": min_type_n, **optimized_details},
            ),
            "calibrated_floor": _policy_summary(
                frame, calibrated_mask, predicted, calibrated_risk,
                {"method": "cross_fitted_calibrated_group_floor",
                 "floor": floor, "min_type_n": min_type_n, **calibrated_details},
            ),
        }
        key = str(int(round(coverage * 100)))
        selection_grid[key] = {
            name: "".join("1" if value else "0" for value in mask)
            for name, mask in masks.items()
        }
        policy_grid[key] = {
            name: {field: summary[field] for field in compact_fields}
            for name, summary in summaries.items()
        }
        for name, mask, risk_values in (
            ("risk_only", risk_mask, scores),
            ("greedy_floor", greedy_mask, scores),
            ("optimized_floor", optimized_mask, scores),
            ("calibrated_floor", calibrated_mask, calibrated_risk),
        ):
            pareto.append({
                "policy": name,
                "target": coverage,
                "actual_coverage": float(mask.mean()),
                "accepted_mae_kg": float(np.abs(predicted[mask] - truth[mask]).mean()),
                "labeled_fuel_share": float(truth[mask].sum() / truth.sum()),
                "mean_predicted_risk_kg": float(risk_values[mask].mean()),
            })
        if math.isclose(coverage, target):
            policies = summaries
            default_masks = masks

    if policies is None or default_masks is None:
        raise RuntimeError("Default target was not evaluated")
    decision_reasons(default_masks["optimized"], default_masks["risk"], types, duration, min_type_n)
    report = {
        "status": "Exploratory historical benchmark dashboard",
        "evaluation_scope": "Cross-fitted exploratory diagnostics on committed September benchmark rows; calibrated batch selection uses evaluation labels indirectly",
        "target": target,
        "floor": floor,
        "min_aircraft_group_n": min_type_n,
        "rows": len(frame),
        "regression": _regression_metrics(predicted, truth),
        "uncertainty": uncertainty,
        "uncertainty_comparison": {
            "global_scaled_conformal": uncertainty,
            "duration_mondrian": duration_mondrian,
            "aircraft_mondrian": aircraft_mondrian,
        },
        "aircraft_held_out_risk_calibration": leave_aircraft_out_risk_calibration(
            scores, predicted, truth, types, min_type_n=min_type_n
        ),
        "natural_adsb_availability": _missingness_reliability(
            frame, predicted, calibrated_risk, interval_half_width
        ),
        "risk_score_sensitivity": _risk_perturbation_sensitivity(
            frame, calibrated_risk, predicted, ties,
            target=target, floor=floor, min_type_n=min_type_n,
        ),
        "group_reliability": _group_reliability(
            frame, predicted, scores, calibrated_risk, interval_half_width, min_type_n
        ),
        "policies": policies,
        "policy_grid": policy_grid,
        "pareto": pareto,
        "selective_curve_summary": _selective_curve_summary(pareto),
        "publication_readiness": {
            "completed": [
                "time-ordered upstream train/risk/calibration/validation design",
                "coverage-constrained and calibrated selective-regression baselines",
                "global and subgroup-conditional cross-fitted uncertainty diagnostics",
                "aircraft-held-out risk-calibrator sensitivity",
                "natural ADS-B availability audit and risk-score perturbation stress test",
                "automated artifact, dashboard and leakage tests",
            ],
            "requires_new_evidence": [
                "untouched post-policy labeled evaluation period",
                "external operator or geography validation",
                "weather and physics-informed model refit on raw trajectories",
                "flight-cluster confirmatory intervals with real flight identifiers",
                "approved live telemetry provider and delayed ground truth",
            ],
        },
    }
    dashboard_payload = {
        "period": payload.get("period", "Committed validation benchmark"),
        "row_schema": list(ROW_COLUMNS[:-1]),
        "rows": [
            [
                row[0], round(float(row[1]), 3), round(float(row[2]), 3),
                round(float(row[3]), 3), round(float(row[4]), 3), int(row[5]),
            ]
            for row in payload["rows"]
        ],
        "calibrated_risk_kg": np.round(calibrated_risk, 3).tolist(),
        "interval_half_width_kg": np.round(interval_half_width, 3).tolist(),
        "selection_grid": selection_grid,
        "report": report,
    }
    return report, dashboard_payload


def render(template: str, payload: dict) -> str:
    marker = "/*__ADVANCED_DATA__*/"
    if template.count(marker) != 1:
        raise ValueError("Expected exactly one advanced-dashboard data marker")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return template.replace(marker, "const DATA=" + data.replace("<", "\\u003c") + ";")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-demo", type=Path, default=Path("demo/index.html"))
    parser.add_argument("--template", type=Path, default=Path("demo/advanced-template.html"))
    parser.add_argument("--dashboard", type=Path, default=Path("demo/advanced.html"))
    parser.add_argument("--report", type=Path, default=Path("reports/advanced-results.json"))
    parser.add_argument("--target", type=float, default=0.75)
    parser.add_argument("--floor", type=float, default=0.5)
    parser.add_argument("--min-type-n", type=int, default=50)
    args = parser.parse_args()
    payload = load_demo_payload(args.source_demo)
    report, dashboard = analyze(
        payload, target=args.target, floor=args.floor, min_type_n=args.min_type_n
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    args.dashboard.parent.mkdir(parents=True, exist_ok=True)
    args.dashboard.write_text(render(args.template.read_text(), dashboard))
    print(f"Wrote {args.report} and {args.dashboard}")


if __name__ == "__main__":
    main()
