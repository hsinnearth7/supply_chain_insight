"""Property-based tests using Hypothesis.

Verifies invariants that must hold for ALL inputs, not just specific examples.
"""

import numpy as np
import pandas as pd
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from app.forecasting.evaluation import ConformalPredictor, mae, mape, rmse
from app.forecasting.models import NaiveMovingAverage

# ---------------------------------------------------------------------------
# Metric Invariant Tests
# ---------------------------------------------------------------------------

class TestMetricInvariants:
    """Properties that must hold for all inputs to metric functions."""

    @given(
        demand=arrays(dtype=float, shape=(50,), elements=st.floats(0.01, 1000)),
        noise=arrays(dtype=float, shape=(50,), elements=st.floats(0.8, 1.2)),
    )
    @settings(max_examples=50)
    def test_mape_non_negative(self, demand, noise):
        """MAPE is always >= 0 for any input."""
        pred = demand * noise
        result = mape(demand, pred)
        assert result >= 0 or np.isnan(result)

    @given(demand=arrays(dtype=float, shape=(50,), elements=st.floats(0.01, 1000)))
    @settings(max_examples=50)
    def test_mape_perfect_is_zero(self, demand):
        """MAPE of perfect predictions is 0."""
        assert mape(demand, demand) == 0.0

    @given(
        demand=arrays(dtype=float, shape=(50,), elements=st.floats(0.01, 1000)),
        noise=arrays(dtype=float, shape=(50,), elements=st.floats(-1, 1)),
    )
    @settings(max_examples=50)
    def test_rmse_non_negative(self, demand, noise):
        """RMSE is always >= 0."""
        pred = demand + noise
        assert rmse(demand, pred) >= 0

    @given(
        demand=arrays(dtype=float, shape=(50,), elements=st.floats(0.01, 1000)),
        noise=arrays(dtype=float, shape=(50,), elements=st.floats(-5, 5)),
    )
    @settings(max_examples=50)
    def test_rmse_geq_mae(self, demand, noise):
        """RMSE >= MAE (Cauchy-Schwarz inequality)."""
        pred = demand + noise
        assert rmse(demand, pred) >= mae(demand, pred) - 1e-10


# ---------------------------------------------------------------------------
# Forecast Model Invariants
# ---------------------------------------------------------------------------

class TestForecastInvariants:
    """Properties that must hold for all forecast models."""

    @given(demand=arrays(dtype=float, shape=(100,), elements=st.floats(0, 1000)))
    @settings(max_examples=20)
    def test_forecast_always_non_negative(self, demand):
        """Any demand input → forecast always >= 0."""
        assume(not np.any(np.isnan(demand)))
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        Y_df = pd.DataFrame({"unique_id": "TEST", "ds": dates, "y": demand})
        model = NaiveMovingAverage(window=30)
        model.fit(Y_df)
        forecasts = model.predict(h=7)
        assert (forecasts["y_hat"] >= 0).all()


# ---------------------------------------------------------------------------
# Conformal Prediction Invariants
# ---------------------------------------------------------------------------

class TestConformalInvariants:
    """Properties of conformal prediction intervals."""

    @given(
        y_true=arrays(dtype=float, shape=(100,), elements=st.floats(1, 1000)),
        noise=arrays(dtype=float, shape=(100,), elements=st.floats(-10, 10)),
    )
    @settings(max_examples=20)
    def test_interval_contains_point_forecast(self, y_true, noise):
        """Confidence interval always contains point forecast."""
        assume(not np.any(np.isnan(y_true)) and not np.any(np.isnan(noise)))
        y_pred = np.maximum(y_true + noise, 0.0)
        cp = ConformalPredictor(target_coverage=0.90)
        cp.calibrate(y_true, y_pred)
        y_lo, y_hi = cp.predict_intervals(y_pred)
        assert (y_lo <= y_pred).all()
        assert (y_hi >= y_pred).all()

    @given(
        y_true=arrays(dtype=float, shape=(100,), elements=st.floats(1, 1000)),
        noise=arrays(dtype=float, shape=(100,), elements=st.floats(-5, 5)),
    )
    @settings(max_examples=20)
    def test_lower_bound_non_negative(self, y_true, noise):
        """Lower bound of prediction interval is always >= 0."""
        assume(not np.any(np.isnan(y_true)) and not np.any(np.isnan(noise)))
        y_pred = np.maximum(y_true + noise, 0.1)
        cp = ConformalPredictor(target_coverage=0.90)
        cp.calibrate(y_true, y_pred)
        y_lo, _ = cp.predict_intervals(y_pred)
        assert (y_lo >= 0).all()
