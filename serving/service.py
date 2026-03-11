"""BentoML service for ChainInsight model serving.

Exposes forecasting ensemble, RL inventory optimization, and drift detection
as production-grade HTTP endpoints with batching and concurrency control.
"""

from __future__ import annotations

import logging
from typing import Any

import bentoml
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@bentoml.service(
    name="chaininsight",
    traffic={
        "timeout": 120,
        "concurrency": 16,
    },
    resources={
        "cpu": "2",
        "memory": "4Gi",
    },
)
class ChainInsightService:
    """Unified model serving for forecasting, RL, and drift detection."""

    def __init__(self) -> None:
        """Load all models on startup."""
        # Forecasting models
        self._forecasters: dict[str, Any] = {}
        self._load_forecasters()

        # RL agent
        self._rl_agent: Any = None
        self._load_rl_agent()

        logger.info("ChainInsightService initialized with all models")

    def _load_forecasters(self) -> None:
        """Load the 6-model forecasting ensemble."""
        model_names = [
            "naive_ma30",
            "sarimax",
            "xgboost",
            "lightgbm",
            "chronos_zs",
            "routing_ensemble",
        ]
        for name in model_names:
            try:
                model_ref = bentoml.models.get(f"chaininsight-{name}:latest")
                self._forecasters[name] = model_ref.load_model()
                logger.info("Loaded forecasting model: %s", name)
            except bentoml.exceptions.NotFound:
                logger.warning("Model %s not found in BentoML store, skipping", name)
            except Exception:
                logger.exception("Failed to load model %s", name)

    def _load_rl_agent(self) -> None:
        """Load PPO inventory optimization agent."""
        try:
            model_ref = bentoml.models.get("chaininsight-ppo-inventory:latest")
            self._rl_agent = model_ref.load_model()
            logger.info("Loaded RL agent: PPO inventory optimizer")
        except bentoml.exceptions.NotFound:
            logger.warning("RL agent not found in BentoML store")
        except Exception:
            logger.exception("Failed to load RL agent")

    def _route_model(self, history: list[float]) -> str:
        """Route to the best model based on history characteristics.

        Routing logic:
        - History < 60 data points -> chronos_zs (cold-start)
        - Intermittent demand (>30% zeros) -> sarimax
        - Otherwise -> lightgbm (best single model)
        """
        arr = np.array(history, dtype=np.float64)

        if len(arr) < 60:
            return "chronos_zs"

        zero_frac = np.sum(arr == 0) / len(arr)
        if zero_frac > 0.3:
            return "sarimax"

        return "lightgbm"

    @bentoml.api()
    def forecast(
        self,
        product_id: str,
        history: list[float],
        horizon: int = 14,
    ) -> dict[str, Any]:
        """Generate demand forecast using the routing ensemble.

        Args:
            product_id: SKU / unique_id for the product.
            history: Historical demand values (chronological order).
            horizon: Forecast horizon in days (default 14).

        Returns:
            Dictionary with forecast values, model used, and confidence intervals.
        """
        if not history:
            return {
                "product_id": product_id,
                "error": "Empty history provided",
                "forecast": [],
            }

        # Route to appropriate model
        model_key = self._route_model(history)

        if model_key not in self._forecasters:
            # Fall back through priority chain
            fallback_order = ["lightgbm", "xgboost", "sarimax", "naive_ma30"]
            model_key = next(
                (k for k in fallback_order if k in self._forecasters),
                None,
            )
            if model_key is None:
                return {
                    "product_id": product_id,
                    "error": "No forecasting models available",
                    "forecast": [],
                }

        model = self._forecasters[model_key]

        try:
            # Build input DataFrame in Nixtla long format
            df = pd.DataFrame({
                "unique_id": product_id,
                "ds": pd.date_range(end=pd.Timestamp.now().normalize(), periods=len(history), freq="D"),
                "y": history,
            })

            # Generate forecast
            if hasattr(model, "predict"):
                forecast_values = model.predict(df, horizon=horizon)
            elif hasattr(model, "forecast"):
                forecast_values = model.forecast(df, h=horizon)
            else:
                # Fallback: simple moving average
                window = min(30, len(history))
                ma = np.mean(history[-window:])
                forecast_values = [float(ma)] * horizon

            # Ensure list output
            if isinstance(forecast_values, (pd.Series, pd.DataFrame)):
                forecast_values = forecast_values.values.flatten().tolist()
            elif isinstance(forecast_values, np.ndarray):
                forecast_values = forecast_values.flatten().tolist()

            # Compute simple confidence intervals (naive +/- 1.96 * std of residuals)
            arr = np.array(history, dtype=np.float64)
            residual_std = float(np.std(arr[-min(30, len(arr)):]))
            lower = [v - 1.96 * residual_std for v in forecast_values]
            upper = [v + 1.96 * residual_std for v in forecast_values]

            return {
                "product_id": product_id,
                "model_used": model_key,
                "horizon": horizon,
                "forecast": forecast_values,
                "confidence_interval": {
                    "lower": lower,
                    "upper": upper,
                    "level": 0.95,
                },
            }

        except Exception as e:
            logger.exception("Forecast failed for %s with model %s", product_id, model_key)
            return {
                "product_id": product_id,
                "error": str(e),
                "model_attempted": model_key,
                "forecast": [],
            }

    @bentoml.api()
    def rl_optimize(
        self,
        state: list[float],
    ) -> dict[str, Any]:
        """Run PPO inventory optimization given current state.

        Args:
            state: Current environment state vector.
                   [inventory_level, demand_forecast, lead_time, holding_cost, ordering_cost]
                   per product (flattened for multi-product).

        Returns:
            Dictionary with recommended actions and expected cost.
        """
        if self._rl_agent is None:
            return {
                "error": "RL agent not loaded",
                "action": [],
            }

        try:
            state_arr = np.array(state, dtype=np.float32)

            # Get action from trained PPO agent
            action, _states = self._rl_agent.predict(state_arr, deterministic=True)

            if isinstance(action, np.ndarray):
                action_list = action.flatten().tolist()
            else:
                action_list = [float(action)]

            return {
                "action": action_list,
                "state_dim": len(state),
                "action_dim": len(action_list),
                "policy": "PPO",
                "deterministic": True,
            }

        except Exception as e:
            logger.exception("RL optimization failed")
            return {
                "error": str(e),
                "action": [],
            }

    @bentoml.api()
    def detect_drift(
        self,
        reference: list[float],
        current: list[float],
    ) -> dict[str, Any]:
        """Detect data or concept drift between reference and current distributions.

        Uses three detection methods:
        - KS test (data drift): threshold 0.05
        - PSI (prediction drift): threshold 0.1
        - MAPE trend (concept drift): flags if degradation > 20%

        Args:
            reference: Reference distribution values (e.g., training data).
            current: Current distribution values (e.g., recent predictions).

        Returns:
            Dictionary with drift detection results per method.
        """
        from scipy import stats

        ref_arr = np.array(reference, dtype=np.float64)
        cur_arr = np.array(current, dtype=np.float64)

        results: dict[str, Any] = {
            "sample_sizes": {
                "reference": len(ref_arr),
                "current": len(cur_arr),
            },
            "drift_detected": False,
            "methods": {},
        }

        # 1. KS Test (data drift)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_arr, cur_arr)
        ks_drift = ks_pvalue < 0.05
        results["methods"]["ks_test"] = {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "threshold": 0.05,
            "drift_detected": ks_drift,
        }

        # 2. PSI (Population Stability Index)
        psi_value = self._compute_psi(ref_arr, cur_arr)
        psi_drift = psi_value > 0.1
        results["methods"]["psi"] = {
            "value": float(psi_value),
            "threshold": 0.1,
            "drift_detected": psi_drift,
            "interpretation": (
                "no_drift" if psi_value < 0.1
                else "moderate_drift" if psi_value < 0.25
                else "significant_drift"
            ),
        }

        # 3. MAPE trend (concept drift proxy)
        if len(ref_arr) == len(cur_arr) and len(ref_arr) > 0:
            mape = float(np.mean(np.abs((ref_arr - cur_arr) / np.where(ref_arr == 0, 1, ref_arr))) * 100)
            mape_drift = mape > 20.0
            results["methods"]["mape_trend"] = {
                "mape_pct": mape,
                "threshold_pct": 20.0,
                "drift_detected": mape_drift,
            }
        else:
            mape_drift = False
            results["methods"]["mape_trend"] = {
                "error": "Reference and current must have equal length for MAPE",
                "drift_detected": False,
            }

        results["drift_detected"] = ks_drift or psi_drift or mape_drift

        return results

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index (PSI)."""
        eps = 1e-6

        # Create bins from reference distribution
        breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi
