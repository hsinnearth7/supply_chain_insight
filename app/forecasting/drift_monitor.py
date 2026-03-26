"""Drift monitoring using Evidently.

Monitors 3 types of drift:
1. Data drift: KS-test on input features (threshold 0.05)
2. Prediction drift: PSI on model outputs (threshold 0.1)
3. Concept drift: MAPE trend over time (threshold 20%)

Schedule: daily check
Trigger: MAPE > 20% for 7 consecutive days → auto retrain pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.log_config import get_logger
from app.settings import get_monitoring_config

logger = get_logger(__name__)


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    drift_type: str
    is_drifted: bool
    statistic: float
    threshold: float
    p_value: float | None = None
    details: dict[str, Any] | None = None


class DriftMonitor:
    """Multi-type drift monitoring for forecasting pipeline.

    Checks data drift, prediction drift, and concept drift.
    Uses Evidently library when available, falls back to manual implementation.
    """

    def __init__(self) -> None:
        config = get_monitoring_config()
        self.ks_threshold = config.get("drift_threshold_ks", 0.05)
        self.psi_threshold = config.get("drift_threshold_psi", 0.1)
        self.mape_threshold = config.get("mape_alert_threshold", 0.20)
        self.retrain_trigger_days = config.get("retrain_trigger_days", 7)
        self._mape_history: list[tuple[pd.Timestamp, float]] = []

    def check_data_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> list[DriftResult]:
        """Check data drift using KS-test on feature distributions.

        Args:
            reference: Reference (training) data.
            current: Current (production) data.
            columns: Columns to check. Defaults to all numeric.

        Returns:
            List of DriftResult per column.
        """
        try:
            return self._check_drift_evidently(reference, current, columns)
        except ImportError:
            return self._check_drift_manual(reference, current, columns)

    def _check_drift_evidently(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None,
    ) -> list[DriftResult]:
        """Use Evidently library for drift detection."""
        from evidently.test_suite import TestSuite
        from evidently.tests import TestColumnDrift

        if columns is None:
            columns = reference.select_dtypes(include=[np.number]).columns.tolist()

        tests = [
            TestColumnDrift(column_name=col, stattest="ks", stattest_threshold=self.ks_threshold)
            for col in columns
            if col in current.columns
        ]

        suite = TestSuite(tests=tests)
        suite.run(reference_data=reference, current_data=current)

        results = []
        for test_result in suite.as_dict().get("tests", []):
            col_name = test_result.get("parameters", {}).get("column_name", "unknown")
            is_drifted = test_result.get("status") == "FAIL"
            stat = test_result.get("parameters", {}).get("stattest_result", {}).get("statistic", 0)
            p_val = test_result.get("parameters", {}).get("stattest_result", {}).get("p_value", 1)

            results.append(DriftResult(
                drift_type="data_drift",
                is_drifted=is_drifted,
                statistic=float(stat),
                threshold=self.ks_threshold,
                p_value=float(p_val),
                details={"column": col_name},
            ))

        return results

    def _check_drift_manual(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None,
    ) -> list[DriftResult]:
        """Manual KS-test drift detection (fallback)."""
        if columns is None:
            columns = reference.select_dtypes(include=[np.number]).columns.tolist()

        results = []
        for col in columns:
            if col not in current.columns or col not in reference.columns:
                continue

            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            stat, p_value = scipy_stats.ks_2samp(ref_vals, cur_vals)

            results.append(DriftResult(
                drift_type="data_drift",
                is_drifted=p_value < self.ks_threshold,
                statistic=float(stat),
                threshold=self.ks_threshold,
                p_value=float(p_value),
                details={"column": col},
            ))

        drifted = sum(1 for r in results if r.is_drifted)
        logger.info("data_drift_check", n_columns=len(results), n_drifted=drifted)
        return results

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> DriftResult:
        """Check prediction drift using PSI (Population Stability Index).

        PSI < 0.1: no drift
        PSI 0.1-0.25: moderate drift
        PSI > 0.25: significant drift
        """
        psi = self._compute_psi(reference_predictions, current_predictions)

        result = DriftResult(
            drift_type="prediction_drift",
            is_drifted=psi > self.psi_threshold,
            statistic=psi,
            threshold=self.psi_threshold,
            details={"psi_interpretation": "no drift" if psi < 0.1 else "moderate" if psi < 0.25 else "significant"},
        )

        logger.info("prediction_drift_check", psi=round(psi, 4), drifted=result.is_drifted)
        return result

    def record_mape(self, timestamp: pd.Timestamp, mape_value: float) -> None:
        """Record daily MAPE for concept drift tracking.

        mape_threshold is a fraction (e.g. 0.20 = 20%).
        record_mape() expects MAPE in percentage form (e.g. 15.0 means 15%).
        """
        self._mape_history.append((timestamp, mape_value))

    def check_concept_drift(self) -> DriftResult:
        """Check concept drift: MAPE > threshold for N consecutive days.

        Trigger: MAPE > 20% for 7 consecutive days → recommend retrain.
        """
        if len(self._mape_history) < self.retrain_trigger_days:
            return DriftResult(
                drift_type="concept_drift",
                is_drifted=False,
                statistic=0,
                threshold=self.mape_threshold,
                details={"reason": "insufficient_history"},
            )

        recent = self._mape_history[-self.retrain_trigger_days:]
        recent_mapes = [m for _, m in recent]
        all_above = all(m > self.mape_threshold * 100 for m in recent_mapes)

        result = DriftResult(
            drift_type="concept_drift",
            is_drifted=all_above,
            statistic=float(np.mean(recent_mapes)),
            threshold=self.mape_threshold * 100,
            details={
                "recent_mapes": [round(m, 2) for m in recent_mapes],
                "consecutive_days_above": sum(1 for m in reversed(recent_mapes) if m > self.mape_threshold * 100),
                "action": "AUTO_RETRAIN" if all_above else "MONITOR",
            },
        )

        if all_above:
            logger.warning(
                "concept_drift_detected",
                avg_mape=round(float(np.mean(recent_mapes)), 2),
                days=self.retrain_trigger_days,
                action="AUTO_RETRAIN",
            )

        return result

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI)."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        if reference.min() == reference.max():
            return 0.0  # Cannot build bins from constant reference

        # Bin boundaries from reference only (current data must not influence bins)
        min_val = float(reference.min())
        max_val = float(reference.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current, bins=bins)[0]

        # Add small epsilon to avoid division by zero
        ref_pcts = (ref_counts + 1e-6) / (ref_counts.sum() + n_bins * 1e-6)
        cur_pcts = (cur_counts + 1e-6) / (cur_counts.sum() + n_bins * 1e-6)

        psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
        return float(psi)

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get summary of all drift monitoring state."""
        return {
            "mape_history_length": len(self._mape_history),
            "latest_mape": self._mape_history[-1][1] if self._mape_history else None,
            "ks_threshold": self.ks_threshold,
            "psi_threshold": self.psi_threshold,
            "mape_threshold": self.mape_threshold,
            "retrain_trigger_days": self.retrain_trigger_days,
        }
