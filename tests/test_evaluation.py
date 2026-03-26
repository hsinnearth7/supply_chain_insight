"""Tests for evaluation framework.

Covers: metrics, statistical tests, conformal prediction, walk-forward CV.
"""

import numpy as np
import pytest

from app.forecasting.evaluation import (
    ConformalPredictor,
    cohens_d,
    confidence_interval,
    mae,
    mape,
    rmse,
    wilcoxon_test,
)

# ---------------------------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_mape_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mape(y, y) == 0.0

    def test_mape_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # (10/100 + 20/200) / 2 = (0.1 + 0.1) / 2 = 0.1 = 10%
        assert abs(mape(y_true, y_pred) - 10.0) < 0.01

    def test_mape_handles_zeros(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 110.0])
        # Should skip zero actuals
        result = mape(y_true, y_pred)
        assert abs(result - 10.0) < 0.01

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])
        # sqrt((1 + 4) / 2) = sqrt(2.5)
        assert abs(rmse(y_true, y_pred) - np.sqrt(2.5)) < 0.001

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 5.0])
        # (1 + 0 + 2) / 3 = 1.0
        assert abs(mae(y_true, y_pred) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

class TestWilcoxonTest:
    def test_identical_distributions(self):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        result = wilcoxon_test(values, values)
        assert not result["significant"]

    def test_clearly_different(self):
        baseline = [20.0, 22.0, 21.0, 23.0, 20.5, 21.5, 22.5]
        improved = [10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5]
        result = wilcoxon_test(baseline, improved)
        assert result["significant"]
        assert result["p_value"] < 0.05

    def test_insufficient_data(self):
        result = wilcoxon_test([1.0, 2.0], [1.5, 2.5])
        assert not result["significant"]

    def test_stars_notation(self):
        baseline = [20.0] * 10
        improved = [10.0] * 10
        result = wilcoxon_test(baseline, improved)
        assert result["stars"] in ["*", "**", "***"]


class TestCohensD:
    def test_identical_groups(self):
        result = cohens_d([10.0, 11.0, 12.0], [10.0, 11.0, 12.0])
        assert result["d"] == 0

    def test_large_effect(self):
        group1 = [10.0, 11.0, 12.0, 10.5, 11.5]
        group2 = [20.0, 21.0, 22.0, 20.5, 21.5]
        result = cohens_d(group1, group2)
        assert result["d"] > 0.8
        assert result["magnitude"] == "L"

    def test_magnitude_labels(self):
        # Small effect
        g1 = [10.0, 10.5, 11.0, 10.2, 10.8]
        g2 = [10.3, 10.8, 11.3, 10.5, 11.1]
        result = cohens_d(g1, g2)
        assert result["magnitude"] in ["S", "M"]


class TestConfidenceInterval:
    def test_single_value(self):
        lo, hi = confidence_interval([5.0])
        assert lo == hi == 5.0

    def test_interval_contains_mean(self):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        lo, hi = confidence_interval(values)
        mean = np.mean(values)
        assert lo <= mean <= hi

    def test_wider_with_more_variance(self):
        tight = [10.0, 10.1, 10.0, 10.1, 10.0]
        wide = [5.0, 15.0, 5.0, 15.0, 5.0]
        lo_t, hi_t = confidence_interval(tight)
        lo_w, hi_w = confidence_interval(wide)
        assert (hi_w - lo_w) > (hi_t - lo_t)


# ---------------------------------------------------------------------------
# Conformal Prediction Tests
# ---------------------------------------------------------------------------

class TestConformalPredictor:
    def test_calibration(self):
        cp = ConformalPredictor(target_coverage=0.90)
        y_true = np.array([10, 11, 12, 9, 13, 8, 14, 7, 15, 6], dtype=float)
        y_pred = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float)
        q = cp.calibrate(y_true, y_pred)
        assert q > 0

    def test_intervals_contain_point(self):
        cp = ConformalPredictor(target_coverage=0.90)
        rng = np.random.default_rng(42)
        y_true = rng.normal(100, 10, 100)
        y_pred = y_true + rng.normal(0, 2, 100)
        cp.calibrate(y_true, y_pred)
        y_lo, y_hi = cp.predict_intervals(y_pred)
        assert (y_lo <= y_pred).all()
        assert (y_hi >= y_pred).all()

    def test_predict_before_calibrate_raises(self):
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="calibrate"):
            cp.predict_intervals(np.array([1.0]))

    def test_non_negative_lower_bound(self):
        cp = ConformalPredictor()
        cp.calibrate(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]))
        y_lo, _ = cp.predict_intervals(np.array([0.5]))
        assert (y_lo >= 0).all()
