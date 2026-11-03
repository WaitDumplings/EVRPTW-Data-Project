"""Empirical similarity/dispersion support gates for fixed-reference neighbors.

Only held-out reference predictions calibrate thresholds.  A configurable 95%
joint retention target describes reference support, not prediction accuracy.
Neither source-unit values nor prediction errors choose these thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HIERARCHIES = [
    ["city", "use_type", "neighbor_branch"],
    ["use_type", "neighbor_branch"],
    ["use_type"],
    [],
]
DEFAULTS = {"quality_target_retention": .95,
            "quality_min_group_size": 30,
            "quality_min_neighbors": 1,
            "quality_dispersion_column": "neighbor_log_spread_90"}


def _number(frame: pd.DataFrame, column: str, default=np.nan):
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _labels(frame: pd.DataFrame, columns):
    return pd.DataFrame({c: (frame[c].fillna("__missing__").astype(str).to_numpy()
                            if c in frame else np.full(len(frame), "__missing__"))
                         for c in columns}, index=frame.index)


def _settings(cfg):
    result = DEFAULTS.copy()
    for key, value in (cfg or {}).items():
        if key in result:
            result[key] = value
    # Short aliases make the helper convenient without dict-key renaming in
    # small standalone callers; the recorded config always uses full names.
    for short, full in [("target_retention", "quality_target_retention"),
                        ("min_group_size", "quality_min_group_size"),
                        ("min_neighbors", "quality_min_neighbors")]:
        if cfg and short in cfg:
            result[full] = cfg[short]
    target = float(result["quality_target_retention"])
    if not 0 < target <= 1:
        raise ValueError("quality_target_retention must be in (0, 1]")
    result["quality_target_retention"] = target
    result["quality_dispersion_column"] = str(result["quality_dispersion_column"])
    for key in ("quality_min_group_size", "quality_min_neighbors"):
        result[key] = int(result[key])
        if result[key] < 1:
            raise ValueError(f"{key} must be positive")
    return result


def _metric_inputs(frame, cfg, include_weights=False):
    cost = _number(frame, "neighbor_cost")
    spread = _number(frame, cfg["quality_dispersion_column"])
    count = _number(frame, "neighbor_count", np.inf)
    valid = (np.isfinite(cost) & (cost >= 0) & np.isfinite(spread) & (spread >= 0)
             & (count >= cfg["quality_min_neighbors"]))
    if "neighbor_units" in frame:
        units = _number(frame, "neighbor_units")
        valid &= np.isfinite(units) & (units >= 1)
    if "neighbor_valid" in frame:
        valid &= frame.neighbor_valid.fillna(False).to_numpy(dtype=bool)
    weights = _number(frame, "sampling_weight", 1.)
    if include_weights:
        valid &= np.isfinite(weights) & (weights > 0)
    return cost, spread, count, weights, valid


def _sorted_quantiles(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=float)
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.
    return values[order], cumulative


def _at_quantile(sorted_values, cumulative, q):
    position = int(np.searchsorted(cumulative, q, side="left"))
    return float(sorted_values[min(position, len(sorted_values)-1)])


def _joint_rectangle(cost, spread, weights, target):
    """Find the smallest common marginal-quantile rectangle retaining target.

    Inclusive empirical order-statistic limits handle point masses such as
    zero dispersion correctly.  Equal marginal quantiles treat cost and dispersion
    symmetrically; their *joint* weighted retention determines the quantile.
    """
    cost_order, cost_cdf = _sorted_quantiles(cost, weights)
    spread_order, spread_cdf = _sorted_quantiles(spread, weights)
    total = weights.sum()

    def evaluate(q):
        c = _at_quantile(cost_order, cost_cdf, q)
        s = _at_quantile(spread_order, spread_cdf, q)
        keep = (cost <= c) & (spread <= s)
        return float(weights[keep].sum()/total), c, s

    retention, c, s = evaluate(target)
    q = target
    if retention + 1e-12 < target:
        low, high = target, 1.
        for _ in range(48):
            middle = (low + high)/2
            if evaluate(middle)[0] + 1e-12 >= target:
                high = middle
            else:
                low = middle
        q = high
        retention, c, s = evaluate(q)
    return {"cost_limit": c, "dispersion_limit": s,
            "common_marginal_quantile": float(q),
            "joint_retention_calibration": retention,
            "unweighted_retention_calibration": float(((cost <= c) & (spread <= s)).mean())}


def _error_diagnostics(frame, accepted):
    if "source_units" not in frame or "neighbor_units" not in frame:
        return None
    source = _number(frame, "source_units")
    pred = _number(frame, "neighbor_units")
    weight = _number(frame, "sampling_weight", 1.)
    valid = (np.isfinite(source) & (source >= 1) & np.isfinite(pred) & (pred >= 1)
             & np.isfinite(weight) & (weight > 0))
    diagnostics = {}
    for name, choice in [("accepted", accepted), ("rejected", ~accepted)]:
        take = valid & choice
        n = int(take.sum())
        result = {"rows": n}
        if n:
            ratio = np.maximum(source[take]/pred[take], pred[take]/source[take])
            result["log_mae"] = float(np.average(np.abs(np.log(source[take]/pred[take])), weights=weight[take]))
            result["within_factor2"] = float(np.average(ratio <= 2., weights=weight[take]))
        diagnostics[name] = result
    return diagnostics


def fit_quality_thresholds(calibration_df: pd.DataFrame, cfg=None):
    """Fit data-based rectangles using calibration predictions only.

    Required metrics: neighbor_cost and the configured log-unit dispersion.
    The default is neighbor_log_spread_90 (weighted log-P90 minus log-P10);
    Frames without this column fall back to neighbor_log_mad at fit.
    The actual fitted metric is recorded and is required when applying gates.
    Stratification uses
    city/use_type/neighbor_branch, with pooled fallbacks.  sampling_weight and
    neighbor_count are optional.  If source_units is available, its errors are
    recorded *after* threshold selection solely as calibration diagnostics.
    """
    config = _settings(cfg)
    requested_dispersion = config["quality_dispersion_column"]
    if requested_dispersion not in calibration_df:
        config["quality_dispersion_column"] = "neighbor_log_mad"
    actual_dispersion = config["quality_dispersion_column"]
    cost, spread, count, weights, valid = _metric_inputs(calibration_df, config, include_weights=True)
    data = calibration_df.loc[valid].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No finite positive-weight held-out reference predictions to calibrate")
    data["_cost"] = cost[valid]
    data["_spread"] = spread[valid]
    data["_weight"] = weights[valid]
    normalized = _labels(data, ["city", "use_type", "neighbor_branch"])
    rows = []
    for columns in HIERARCHIES:
        if columns:
            keys = normalized[columns]
            groups = keys.groupby(columns, dropna=False, sort=True).groups.items()
        else:
            groups = [((), data.index)]
        for key, positions in groups:
            if not isinstance(key, tuple):
                key = (key,)
            group = data.loc[positions]
            if columns and len(group) < config["quality_min_group_size"]:
                continue
            rectangle = _joint_rectangle(group._cost.to_numpy(), group._spread.to_numpy(),
                                         group._weight.to_numpy(), config["quality_target_retention"])
            accepted = ((group._cost.to_numpy() <= rectangle["cost_limit"]) &
                        (group._spread.to_numpy() <= rectangle["dispersion_limit"]))
            row = {"columns": list(columns), "key": list(key), "calibration_rows": len(group),
                   "calibration_weight": float(group._weight.sum()), **rectangle}
            error = _error_diagnostics(group, accepted)
            if error is not None:
                row["calibration_error_diagnostics"] = error
            rows.append(row)
    return {"schema": "empirical_neighbor_reference_support_v2", "config": config,
            "dispersion_requested_column": requested_dispersion,
            "dispersion_column": actual_dispersion,
            "calibration_rows_total": len(calibration_df), "calibration_rows_valid": len(data),
            "calibration_rows_invalid": int((~valid).sum()),
            "hierarchies": HIERARCHIES, "strata": rows,
            "method": f"Smallest common-marginal-quantile cost/{actual_dispersion} rectangle reaching weighted joint calibration retention",
            "interpretation": "Reference-support retention, not an accuracy guarantee; test labels were not read",
            "threshold_selection_uses_source_units": False}


def apply_quality(pred_df: pd.DataFrame, thresholds):
    """Attach thresholds and gate outcomes, preserving every input row/index."""
    config = _settings(thresholds.get("config"))
    if "dispersion_column" in thresholds:
        config["quality_dispersion_column"] = thresholds["dispersion_column"]
    elif thresholds.get("schema") == "empirical_neighbor_reference_support_v1":
        config["quality_dispersion_column"] = "neighbor_log_mad"
    n = len(pred_df)
    labels = _labels(pred_df, ["city", "use_type", "neighbor_branch"])
    limits = {"quality_cost_limit": np.full(n, np.nan),
              "quality_dispersion_limit": np.full(n, np.nan),
              "quality_calibration_count": np.zeros(n, dtype=np.int64),
              "quality_calibration_retention": np.full(n, np.nan)}
    strata = np.full(n, "unavailable", dtype=object)
    assigned = np.zeros(n, dtype=bool)
    for columns in thresholds.get("hierarchies", HIERARCHIES):
        group_rows = [r for r in thresholds["strata"] if r["columns"] == columns]
        if not group_rows:
            continue
        if not columns:
            position = np.zeros(n, dtype=np.int64)
        elif len(columns) == 1:
            index = pd.Index([r["key"][0] for r in group_rows])
            position = index.get_indexer(labels[columns[0]])
        else:
            index = pd.MultiIndex.from_tuples([tuple(r["key"]) for r in group_rows], names=columns)
            position = index.get_indexer(pd.MultiIndex.from_frame(labels[columns]))
        take = ~assigned & (position >= 0)
        chosen = position[take]
        for output, source in [("quality_cost_limit", "cost_limit"),
                               ("quality_dispersion_limit", "dispersion_limit"),
                               ("quality_calibration_count", "calibration_rows"),
                               ("quality_calibration_retention", "joint_retention_calibration")]:
            values = np.asarray([r[source] for r in group_rows])
            limits[output][take] = values[chosen]
        names = np.asarray(["/".join(columns) + ":" + "|".join(r["key"])
                            if columns else "global" for r in group_rows], dtype=object)
        strata[take] = names[chosen]
        assigned[take] = True
    cost, spread, count, _, valid = _metric_inputs(pred_df, config)
    result = pred_df.copy()
    for column, values in limits.items():
        result[column] = values
    result["quality_stratum"] = strata
    result["quality_dispersion_metric"] = config["quality_dispersion_column"]
    result["quality_inputs_valid"] = valid
    result["quality_support_count_pass"] = count >= config["quality_min_neighbors"]
    result["similarity_pass"] = valid & assigned & (cost <= limits["quality_cost_limit"])
    result["dispersion_pass"] = valid & assigned & (spread <= limits["quality_dispersion_limit"])
    result["quality_pass"] = result.similarity_pass & result.dispersion_pass
    return result
