"""Evaluation framework: Walk-Forward CV + Statistical Tests + Conformal Prediction.

Implements:
- 12-fold walk-forward cross-validation (monthly retrain, 1-month horizon)
- Wilcoxon signed-rank test (vs Naive baseline, α=0.05)
- Cohen's d effect size (S<0.5, M 0.5-0.8, L>0.8)
- Conformal prediction for calibrated confidence intervals
- Ablation study framework
- Routing threshold sensitivity analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.forecasting.models import ForecastModel
from app.log_config import get_logger
from app.settings import get_eval_config

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_handling: str = "mask",
    epsilon: float = 1.0,
) -> float:
    """Mean Absolute Percentage Error (handles zeros).

    Zero-handling behavior:
      - 'mask' (default): excludes zero actuals from the calculation entirely.
        This inflates accuracy when many true values are zero because those
        (potentially large) forecast errors are silently dropped.
      - 'epsilon': replaces zero actuals with ``epsilon`` so that every
        observation contributes to the error metric.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        zero_handling: Strategy for zero actuals — 'mask' or 'epsilon'.
        epsilon: Replacement value when zero_handling='epsilon'.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5%).
    """
    if zero_handling == "epsilon":
        mask = np.ones(len(y_true), dtype=bool)
        y_true_safe = np.where(y_true == 0, epsilon, y_true)
    else:
        mask = y_true > 0
        y_true_safe = y_true
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs((y_true_safe[mask] - y_pred[mask]) / y_true_safe[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def coverage(y_true: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray) -> float:
    """Prediction interval coverage."""
    within = (y_true >= y_lo) & (y_true <= y_hi)
    return float(within.mean())


# ---------------------------------------------------------------------------
# Walk-Forward Cross-Validation
# ---------------------------------------------------------------------------


@dataclass
class CVFoldResult:
    """Result of a single cross-validation fold."""

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict[str, float]
    n_train: int
    n_test: int


def walk_forward_cv(
    model: ForecastModel,
    Y_df: pd.DataFrame,
    X_df: pd.DataFrame | None = None,
    n_windows: int = 12,
    step_size: int = 30,
    horizon: int = 14,
) -> list[CVFoldResult]:
    """Walk-forward cross-validation with monthly retrain.

    Args:
        model: ForecastModel to evaluate.
        Y_df: Full dataset in Nixtla format (unique_id, ds, y).
        X_df: Optional exogenous features.
        n_windows: Number of CV folds.
        step_size: Days between successive training windows.
        horizon: Forecast horizon in days.

    Returns:
        List of CVFoldResult for each fold.
    """
    config = get_eval_config()
    n_windows = config.get("cv_windows", n_windows)
    step_size = config.get("step_size", step_size)
    horizon = config.get("horizon", horizon)

    dates = sorted(Y_df["ds"].unique())
    total_days = len(dates)

    results = []
    for fold in range(n_windows):
        # Compute split point
        test_end_idx = total_days - 1 - fold * step_size
        test_start_idx = test_end_idx - horizon + 1
        train_end_idx = test_start_idx - 1

        if train_end_idx < 60:  # Need minimum 60 days for training
            break

        train_end = dates[train_end_idx]
        test_start = dates[test_start_idx]
        test_end = dates[test_end_idx]

        # Split data
        Y_train = Y_df[Y_df["ds"] <= train_end]
        Y_test = Y_df[(Y_df["ds"] >= test_start) & (Y_df["ds"] <= test_end)]

        if len(Y_test) == 0:
            continue

        X_train = None
        if X_df is not None:
            X_train = X_df[X_df["ds"] <= train_end]

        # Fit and predict
        model.fit(Y_train, X_train)
        forecasts = model.predict(h=horizon)

        # Merge actuals with forecasts
        merged = Y_test.merge(forecasts, on=["unique_id", "ds"], how="inner")

        if len(merged) == 0:
            continue

        y_true = merged["y"].values
        y_pred = merged["y_hat"].values

        fold_metrics = {
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
        }

        results.append(
            CVFoldResult(
                fold=fold,
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                metrics=fold_metrics,
                n_train=len(Y_train),
                n_test=len(Y_test),
            )
        )

        logger.info(
            "cv_fold_complete",
            model=model.name,
            fold=fold,
            mape=round(fold_metrics["mape"], 2),
        )

    return results


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------


def wilcoxon_test(
    baseline_mapes: list[float],
    model_mapes: list[float],
) -> dict[str, Any]:
    """Wilcoxon signed-rank test comparing model vs baseline.

    Args:
        baseline_mapes: Per-fold MAPE from baseline model.
        model_mapes: Per-fold MAPE from comparison model.

    Returns:
        Dict with statistic, p_value, significant flag, and stars.
    """
    if len(baseline_mapes) != len(model_mapes) or len(baseline_mapes) < 5:
        return {"statistic": None, "p_value": None, "significant": False, "stars": ""}

    diffs = np.array(baseline_mapes) - np.array(model_mapes)
    if np.all(diffs == 0):
        return {"statistic": 0, "p_value": 1.0, "significant": False, "stars": ""}

    stat, p_value = scipy_stats.wilcoxon(diffs, alternative="greater")

    stars = ""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "stars": stars,
    }


def cohens_d(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """Cohen's d effect size.

    Positive d means group1 > group2 (baseline worse than model = improvement).
    The sign is preserved to indicate directionality.

    Returns:
        Dict with d value, magnitude (S/M/L), and interpretation.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return {"d": 0, "magnitude": "N/A", "interpretation": "Insufficient data"}

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return {"d": 0, "magnitude": "N/A", "interpretation": "Zero variance"}

    d = float((mean1 - mean2) / pooled_std)

    abs_d = abs(d)
    if abs_d < 0.5:
        magnitude = "S"
        interpretation = "Small effect"
    elif abs_d < 0.8:
        magnitude = "M"
        interpretation = "Medium effect"
    else:
        magnitude = "L"
        interpretation = "Large effect"

    return {"d": round(d, 2), "magnitude": magnitude, "interpretation": interpretation}


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Compute confidence interval for a list of values."""
    n = len(values)
    if n < 2:
        return (float('nan'), float('nan'))

    mean = np.mean(values)
    se = scipy_stats.sem(values)
    h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return (round(float(mean - h), 2), round(float(mean + h), 2))


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """Split conformal prediction for calibrated confidence intervals.

    Target coverage: 90% (configurable).
    Method: compute residuals on calibration set, use quantile as interval width.
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self._quantile: float | None = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calibrate on held-out calibration set.

        Args:
            y_true: Actual values from calibration set.
            y_pred: Point forecasts from calibration set.

        Returns:
            Calibrated quantile value.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            self._quantile = 0.0
            return 0.0
        residuals = np.abs(y_true - y_pred)
        alpha = 1 - self.target_coverage
        # Finite-sample correction
        n = len(residuals)
        adjusted_quantile = min(1.0, (1 - alpha) * (1 + 1 / n))
        self._quantile = float(np.quantile(residuals, adjusted_quantile))
        logger.info("conformal_calibrated", quantile=self._quantile, n_calibration=n)
        return self._quantile

    def predict_intervals(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals.

        Args:
            y_pred: Point forecasts.

        Returns:
            (y_lo, y_hi) arrays.
        """
        if self._quantile is None:
            raise ValueError("Must call calibrate() before predict_intervals()")

        # TODO: For proportional intervals, calibrate on relative residuals instead
        y_lo = np.maximum(0, y_pred - self._quantile)
        y_hi = y_pred + self._quantile
        return y_lo, y_hi


# ---------------------------------------------------------------------------
# Benchmark Table Builder
# ---------------------------------------------------------------------------


@dataclass
class ModelBenchmark:
    """Complete benchmark result for a single model."""

    model_name: str
    mape_mean: float
    mape_ci: tuple[float, float]
    vs_baseline: float | None
    p_value: float | None
    cohens_d: float | None
    cohens_d_magnitude: str | None
    best_for: str
    per_fold_mapes: list[float]


def build_benchmark_table(
    models: dict[str, ForecastModel],
    Y_df: pd.DataFrame,
    baseline_name: str = "naive_ma30",
) -> list[ModelBenchmark]:
    """Run all models through walk-forward CV and build comparison table.

    Args:
        models: Dict of model_name -> ForecastModel.
        Y_df: Full dataset in Nixtla format.
        baseline_name: Name of baseline model for statistical comparison.

    Returns:
        List of ModelBenchmark results, sorted by MAPE.
    """
    all_results: dict[str, list[CVFoldResult]] = {}
    for name, model in models.items():
        logger.info("benchmark_start", model=name)
        cv_results = walk_forward_cv(model, Y_df)
        all_results[name] = cv_results

    # Extract per-fold MAPEs
    fold_mapes: dict[str, list[float]] = {}
    for name, results in all_results.items():
        fold_mapes[name] = [r.metrics["mape"] for r in results]

    baseline_mapes = fold_mapes.get(baseline_name, [])

    best_for_map = {
        "naive_ma30": "Reference",
        "sarimax": "Seasonal / cold-start",
        "xgboost": "Feature interactions",
        "lightgbm": "Best overall",
        "chronos2_zs": "Cold-start / no history",
        "routing_ensemble": "Mixed best",
    }

    benchmarks = []
    for name, mapes in fold_mapes.items():
        if not mapes:
            continue

        mean_mape = round(float(np.nanmean(mapes)), 1)
        ci = confidence_interval(mapes)

        vs_baseline = None
        p_val = None
        cd = None
        cd_mag = None

        if name != baseline_name and baseline_mapes:
            vs_baseline = round(mean_mape - float(np.nanmean(baseline_mapes)), 1)
            test_result = wilcoxon_test(baseline_mapes, mapes)
            p_val = test_result["p_value"]
            d_result = cohens_d(baseline_mapes, mapes)
            cd = d_result["d"]
            cd_mag = d_result["magnitude"]

        benchmarks.append(
            ModelBenchmark(
                model_name=name,
                mape_mean=mean_mape,
                mape_ci=ci,
                vs_baseline=vs_baseline,
                p_value=p_val,
                cohens_d=cd,
                cohens_d_magnitude=cd_mag,
                best_for=best_for_map.get(name, ""),
                per_fold_mapes=mapes,
            )
        )

    benchmarks.sort(key=lambda b: b.mape_mean if not np.isnan(b.mape_mean) else float('inf'))
    return benchmarks


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------


def run_ablation_study(
    model: ForecastModel,
    Y_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Systematic feature ablation study.

    Removes one feature group at a time and measures MAPE impact.

    Args:
        model: Tree-based model with feature importance.
        Y_df: Full dataset.
        feature_groups: Dict of group_name -> list of feature column names.

    Returns:
        List of dicts with ablation results.
    """
    # NOT IMPLEMENTED: This is a placeholder stub. The per-group ablation loop
    # does not actually retrain the model with features removed — it only records
    # skeleton entries with None values. Do not rely on these results.
    import logging
    logging.getLogger(__name__).warning("Ablation study is a placeholder — results are not real")

    # Full model baseline
    full_results = walk_forward_cv(model, Y_df)
    full_mapes = [r.metrics["mape"] for r in full_results]
    full_mean = float(np.mean(full_mapes))

    ablation_results = [
        {"config": "Full model", "mape": round(full_mean, 1), "delta": 0, "p_value": None, "note": "All features"}
    ]

    for group_name, feature_cols in feature_groups.items():
        # Note: actual ablation requires modifying the model's feature set
        # This is a placeholder framework — implementation depends on model internals
        logger.info("ablation_start", removed_group=group_name, features=feature_cols)
        # In practice, rebuild model without these features and re-evaluate
        ablation_results.append(
            {
                "config": f"− {group_name}",
                "mape": None,  # To be filled by actual ablation
                "delta": None,
                "p_value": None,
                "note": f"Without {', '.join(feature_cols)}",
            }
        )

    return ablation_results


# ---------------------------------------------------------------------------
# Routing Threshold Sensitivity
# ---------------------------------------------------------------------------


def routing_threshold_sensitivity(
    Y_df: pd.DataFrame,
    thresholds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Analyze sensitivity of routing ensemble to cold-start threshold.

    Args:
        Y_df: Full dataset.
        thresholds: List of threshold values to test.

    Returns:
        List of dicts with threshold -> MAPE mapping.
    """
    from app.forecasting.models import RoutingEnsemble

    if thresholds is None:
        thresholds = [30, 40, 50, 60, 70, 90, 120]

    results = []
    for threshold in thresholds:
        model = RoutingEnsemble(cold_start_threshold_days=threshold)
        cv_results = walk_forward_cv(model, Y_df)
        mapes = [r.metrics["mape"] for r in cv_results]
        mean_mape = round(float(np.nanmean(mapes)), 1) if mapes else None
        results.append({"threshold_days": threshold, "mape": mean_mape})
        logger.info("threshold_sensitivity", threshold=threshold, mape=mean_mape)

    return results
