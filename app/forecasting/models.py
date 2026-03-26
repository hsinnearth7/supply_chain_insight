"""Forecasting model zoo with unified fit/predict interface (Strategy pattern).

All models implement the ForecastModel protocol:
    - fit(Y_train, X_train=None) → self
    - predict(h, X_future=None) → pd.DataFrame with (unique_id, ds, y_hat)
    - name → str

Models:
    1. NaiveMovingAverage   — rolling mean baseline
    2. SARIMAXForecaster    — seasonal ARIMA with exogenous
    3. XGBoostForecaster    — gradient boosting with lag features
    4. LightGBMForecaster   — LightGBM with lag features
    5. ChronosForecaster    — Amazon Chronos-2 zero-shot (foundation model)
    6. RoutingEnsemble      — cold-start routing logic
    7. ProphetForecaster    — Facebook Prophet (fallback: seasonal decompose)
    8. LSTMForecaster       — LSTM neural network (fallback: weighted rolling avg)
    9. NBEATSForecaster     — N-BEATS neural architecture (fallback: polynomial + Fourier)
   10. TFTForecaster        — Temporal Fusion Transformer (fallback: feature-weighted regression)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from app.log_config import get_logger
from app.settings import get_model_config

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------


class ForecastModel(ABC):
    """Unified forecast model interface (Strategy pattern)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""

    @abstractmethod
    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "ForecastModel":
        """Fit model on training data.

        Args:
            Y_train: Nixtla format (unique_id, ds, y).
            X_train: Optional exogenous features (unique_id, ds, ...).

        Returns:
            self for chaining.
        """

    @abstractmethod
    def predict(
        self,
        h: int,
        X_future: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            h: Forecast horizon (number of periods ahead).
            X_future: Optional future exogenous features.

        Returns:
            DataFrame with columns (unique_id, ds, y_hat).
        """


# ---------------------------------------------------------------------------
# 1. Naive Moving Average
# ---------------------------------------------------------------------------


class NaiveMovingAverage(ForecastModel):
    """Simple moving average baseline."""

    def __init__(self, window: int = 30):
        self.window = window
        self._last_values: dict[str, np.ndarray] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    @property
    def name(self) -> str:
        return f"naive_ma{self.window}"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "NaiveMovingAverage":
        for uid, group in Y_train.groupby("unique_id"):
            group = group.sort_values("ds")
            values = group["y"].values
            self._last_values[str(uid)] = values[-self.window :]
            self._last_dates[str(uid)] = group["ds"].max()
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        records = []
        for uid, values in self._last_values.items():
            ma = float(np.mean(values))
            last_ds = self._last_dates.get(uid, pd.Timestamp("2024-01-01"))
            for step in range(h):
                records.append({"unique_id": uid, "ds": last_ds + pd.Timedelta(days=step + 1), "y_hat": max(0, ma)})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. SARIMAX
# ---------------------------------------------------------------------------


class SARIMAXForecaster(ForecastModel):
    """Seasonal ARIMA with exogenous variables.

    Uses statsforecast for fast fitting when available, falls back to statsmodels.
    Good for: cold-start SKUs, seasonal patterns, intermittent demand.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self._models: dict[str, Any] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    @property
    def name(self) -> str:
        return "sarimax"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "SARIMAXForecaster":
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA

            sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D", n_jobs=1)
            sf.fit(Y_train)
            self._sf = sf
            self._use_statsforecast = True
        except ImportError:
            self._use_statsforecast = False
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values
                if len(y) < 14:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(
                            y, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                        )
                        fitted = model.fit(disp=False, maxiter=50)
                        self._models[str(uid)] = fitted
                        self._last_dates[str(uid)] = grp["ds"].iloc[-1]
                except Exception:
                    logger.warning("sarimax_fit_failed", uid=uid)
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if getattr(self, "_use_statsforecast", False):
            forecasts = self._sf.predict(h=h)
            forecasts = forecasts.reset_index()
            forecasts = forecasts.rename(columns={"AutoARIMA": "y_hat"})
            forecasts["y_hat"] = forecasts["y_hat"].clip(lower=0)
            return forecasts[["unique_id", "ds", "y_hat"]]

        records = []
        for uid, model in self._models.items():
            try:
                forecast = model.forecast(steps=h)
                last_ds = self._last_dates[uid]
                future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")
                for ds, yhat in zip(future_dates, forecast):
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
            except Exception:
                logger.warning("sarimax_predict_failed", uid=uid)
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. XGBoost Forecaster
# ---------------------------------------------------------------------------


def _build_lag_features(Y_df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Build lag and rolling features for tree-based models."""
    if lags is None:
        lags = [1, 7, 14, 28]

    frames = []
    for uid, grp in Y_df.groupby("unique_id"):
        grp = grp.sort_values("ds").copy()
        for lag in lags:
            grp[f"lag_{lag}"] = grp["y"].shift(lag)
        grp["rolling_mean_7"] = grp["y"].shift(1).rolling(7, min_periods=1).mean()
        grp["rolling_std_7"] = grp["y"].shift(1).rolling(7, min_periods=1).std().fillna(0)
        grp["rolling_mean_28"] = grp["y"].shift(1).rolling(28, min_periods=1).mean()
        grp["day_of_week"] = grp["ds"].dt.dayofweek
        grp["month"] = grp["ds"].dt.month
        grp["day_of_year"] = grp["ds"].dt.dayofyear
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    return result


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_28",
    "day_of_week", "month", "day_of_year",
]


def _compute_features(history: list[float], ds: pd.Timestamp) -> list[float]:
    """Compute lag features for a single prediction step.

    Shared by XGBoostForecaster and LightGBMForecaster.
    """
    n = len(history)
    lag_1 = history[-1] if n >= 1 else 0
    lag_7 = history[-7] if n >= 7 else lag_1
    lag_14 = history[-14] if n >= 14 else lag_1
    lag_28 = history[-28] if n >= 28 else lag_1
    # Match training's shift(1).rolling(W) — exclude current value (history[-1])
    rm7 = float(np.mean(history[-8:-1])) if n >= 8 else (float(np.mean(history[:-1])) if n >= 2 else 0.0)
    rs7 = float(np.std(history[-8:-1], ddof=1)) if n >= 8 else 0.0
    rm28 = float(np.mean(history[-29:-1])) if n >= 29 else (float(np.mean(history[:-1])) if n >= 2 else 0.0)
    return [lag_1, lag_7, lag_14, lag_28, rm7, rs7, rm28, ds.dayofweek, ds.month, ds.dayofyear]


class XGBoostForecaster(ForecastModel):
    """XGBoost time series forecaster with lag features.

    Good for: feature interactions, non-linear patterns, mature SKUs.
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 5, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._model = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "XGBoostForecaster":
        from xgboost import XGBRegressor

        featured = _build_lag_features(Y_train)
        featured = featured.dropna(subset=FEATURE_COLS)

        X = featured[FEATURE_COLS].values
        y = featured["y"].values

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=1,
        )
        self._model.fit(X, y)
        self._train_data = Y_train
        logger.info("model_fitted", model="xgboost", n_samples=len(X))
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._model is None or self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            history = grp["y"].values.tolist()
            last_ds = grp["ds"].iloc[-1]

            for step in range(h):
                ds = last_ds + pd.Timedelta(days=step + 1)
                features = _compute_features(history, ds)
                yhat = float(self._model.predict(np.array([features]))[0])
                yhat = max(0, yhat)
                records.append({"unique_id": uid, "ds": ds, "y_hat": yhat})
                history.append(yhat)

        return pd.DataFrame(records)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Return feature importances if model is fitted."""
        if self._model is None:
            return None
        return dict(zip(FEATURE_COLS, self._model.feature_importances_))


# ---------------------------------------------------------------------------
# 4. LightGBM Forecaster
# ---------------------------------------------------------------------------


class LightGBMForecaster(ForecastModel):
    """LightGBM time series forecaster with lag features.

    Best overall MAPE for mature SKUs. Fast training and inference.
    """

    def __init__(self, n_estimators: int = 300, num_leaves: int = 31, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self._model = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "LightGBMForecaster":
        import lightgbm as lgb

        featured = _build_lag_features(Y_train)
        featured = featured.dropna(subset=FEATURE_COLS)

        X = featured[FEATURE_COLS].values
        y = featured["y"].values

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        self._model.fit(X, y)
        self._train_data = Y_train
        logger.info("model_fitted", model="lightgbm", n_samples=len(X))
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._model is None or self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            history = grp["y"].values.tolist()
            last_ds = grp["ds"].iloc[-1]

            for step in range(h):
                ds = last_ds + pd.Timedelta(days=step + 1)
                features = _compute_features(history, ds)
                yhat = float(self._model.predict(np.array([features]))[0])
                yhat = max(0, yhat)
                records.append({"unique_id": uid, "ds": ds, "y_hat": yhat})
                history.append(yhat)

        return pd.DataFrame(records)

    @property
    def feature_importance(self) -> dict[str, float] | None:
        if self._model is None:
            return None
        return dict(zip(FEATURE_COLS, self._model.feature_importances_))


# ---------------------------------------------------------------------------
# 5. Chronos-2 Zero-Shot Forecaster
# ---------------------------------------------------------------------------


class ChronosForecaster(ForecastModel):
    """Amazon Chronos-2 foundation model for zero-shot forecasting.

    No training needed — uses pretrained time series knowledge.
    Good for: cold-start SKUs, benchmark baseline.
    """

    def __init__(self, model_name: str = "amazon/chronos-t5-small", prediction_length: int = 14):
        self.model_name = model_name
        self.prediction_length = prediction_length
        self._pipeline = None
        self._train_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "chronos2_zs"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "ChronosForecaster":
        """Chronos is zero-shot — fit just stores data for predict."""
        self._train_data = Y_train
        try:
            import torch
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("chronos_loaded", model=self.model_name)
        except ImportError:
            logger.warning("chronos_not_installed", fallback="naive_ma30")
            self._pipeline = None
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        if h > self.prediction_length:
            import warnings
            warnings.warn(f"Requested horizon {h} exceeds prediction_length {self.prediction_length}, truncating to {self.prediction_length}")
        h = min(h, self.prediction_length)

        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if self._pipeline is not None:
                import torch

                context = torch.tensor(grp["y"].values[-512:], dtype=torch.float32).unsqueeze(0)
                forecast = self._pipeline.predict(context, prediction_length=h)
                median = forecast.median(dim=1).values.squeeze().numpy()
                for ds, yhat in zip(future_dates, median):
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
            else:
                # Fallback: simple moving average
                ma = float(grp["y"].tail(30).mean())
                for ds in future_dates:
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, ma)})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Routing Ensemble
# ---------------------------------------------------------------------------


class RoutingEnsemble(ForecastModel):
    """Intelligent routing ensemble based on SKU characteristics.

    Routing logic:
        - history < threshold_days → Chronos-2 zero-shot (cold start)
        - intermittency > 0.5      → SARIMAX (handles zeros well)
        - otherwise                → LightGBM (best overall MAPE)
    """

    def __init__(
        self,
        cold_start_threshold_days: int = 60,
        intermittency_threshold: float = 0.5,
    ):
        self.cold_start_threshold_days = cold_start_threshold_days
        self.intermittency_threshold = intermittency_threshold
        self._models: dict[str, ForecastModel] = {}
        self._routing_decisions: dict[str, str] = {}
        self._sku_stats: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "routing_ensemble"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "RoutingEnsemble":
        # Compute per-SKU statistics for routing
        for uid, grp in Y_train.groupby("unique_id"):
            n_days = len(grp)
            intermittency = float((grp["y"] == 0).mean())
            self._sku_stats[str(uid)] = {
                "n_days": n_days,
                "intermittency": intermittency,
            }

            # Routing decision
            if n_days < self.cold_start_threshold_days:
                self._routing_decisions[str(uid)] = "chronos2_zs"
            elif intermittency > self.intermittency_threshold:
                self._routing_decisions[str(uid)] = "sarimax"
            else:
                self._routing_decisions[str(uid)] = "lightgbm"

        # Determine which models are needed
        needed_models = set(self._routing_decisions.values())
        logger.info(
            "routing_decisions",
            total_skus=len(self._routing_decisions),
            cold_start=sum(1 for v in self._routing_decisions.values() if v == "chronos2_zs"),
            intermittent=sum(1 for v in self._routing_decisions.values() if v == "sarimax"),
            mature=sum(1 for v in self._routing_decisions.values() if v == "lightgbm"),
        )

        # Fit each needed model on its assigned SKUs
        model_config = get_model_config()

        for model_name in needed_models:
            assigned_uids = {uid for uid, m in self._routing_decisions.items() if m == model_name}
            subset = Y_train[Y_train["unique_id"].isin(assigned_uids)]

            if model_name == "chronos2_zs":
                cfg = model_config.get("chronos", {})
                model = ChronosForecaster(
                    model_name=cfg.get("model_name", "amazon/chronos-t5-small"),
                    prediction_length=cfg.get("prediction_length", 14),
                )
            elif model_name == "sarimax":
                cfg = model_config.get("sarimax", {})
                model = SARIMAXForecaster(
                    order=tuple(cfg.get("order", [1, 1, 1])),
                    seasonal_order=tuple(cfg.get("seasonal_order", [1, 1, 1, 7])),
                )
            else:  # lightgbm
                cfg = model_config.get("lightgbm", {})
                model = LightGBMForecaster(
                    n_estimators=cfg.get("n_estimators", 300),
                    num_leaves=cfg.get("num_leaves", 31),
                    learning_rate=cfg.get("learning_rate", 0.1),
                )

            model.fit(subset)
            self._models[model_name] = model

        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        all_forecasts = []
        for model_name, model in self._models.items():
            forecasts = model.predict(h=h, X_future=X_future)
            forecasts["model"] = model_name
            all_forecasts.append(forecasts)

        if not all_forecasts:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat", "model"])

        return pd.concat(all_forecasts, ignore_index=True)

    @property
    def routing_summary(self) -> dict[str, int]:
        """Summary of routing decisions."""
        summary: dict[str, int] = {}
        for model_name in self._routing_decisions.values():
            summary[model_name] = summary.get(model_name, 0) + 1
        return summary


# ---------------------------------------------------------------------------
# 7. Prophet Forecaster
# ---------------------------------------------------------------------------


class ProphetForecaster(ForecastModel):
    """Facebook Prophet forecaster with automatic seasonality detection.

    Falls back to statsmodels seasonal_decompose-based approach when
    the ``prophet`` library is not installed.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._models: dict[str, Any] = {}
        self._train_data: pd.DataFrame | None = None
        self._use_prophet: bool = False

    @property
    def name(self) -> str:
        return "prophet"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "ProphetForecaster":
        self._train_data = Y_train
        try:
            from prophet import Prophet  # type: ignore[import-untyped]

            self._use_prophet = True
            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                prophet_df = grp[["ds", "y"]].copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        changepoint_prior_scale=self.changepoint_prior_scale,
                    )
                    m.fit(prophet_df)
                    self._models[str(uid)] = m
            logger.info("model_fitted", model="prophet", n_series=len(self._models))
        except ImportError:
            logger.warning("prophet_not_installed", fallback="seasonal_decompose")
            self._use_prophet = False
            # Fallback: compute trend + seasonal component via statsmodels
            from statsmodels.tsa.seasonal import seasonal_decompose

            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values
                period = min(7, len(y) // 2) if len(y) >= 4 else 2
                try:
                    decomp = seasonal_decompose(y, model="additive", period=period, extrapolate_trend="freq")
                    self._models[str(uid)] = {
                        "trend_last": float(decomp.trend[~np.isnan(decomp.trend)][-1]),
                        "trend_slope": float(np.polyfit(range(len(y)), y, 1)[0]),
                        "seasonal": decomp.seasonal[-period:].tolist(),
                        "period": period,
                    }
                except Exception:
                    self._models[str(uid)] = {
                        "trend_last": float(np.mean(y[-30:])),
                        "trend_slope": 0.0,
                        "seasonal": [0.0],
                        "period": 1,
                    }
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if self._use_prophet and str(uid) in self._models:
                future_df = pd.DataFrame({"ds": future_dates})
                forecast = self._models[str(uid)].predict(future_df)
                for ds, yhat in zip(future_dates, forecast["yhat"].values):
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
            elif str(uid) in self._models:
                info = self._models[str(uid)]
                for step, ds in enumerate(future_dates):
                    seasonal_val = info["seasonal"][step % info["period"]]
                    yhat = info["trend_last"] + info["trend_slope"] * (step + 1) + seasonal_val
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 8. LSTM Forecaster
# ---------------------------------------------------------------------------


class LSTMForecaster(ForecastModel):
    """LSTM recurrent neural network forecaster.

    Falls back to exponentially-weighted rolling average when PyTorch
    is not installed.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        lookback: int = 30,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lookback = lookback
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._models: dict[str, Any] = {}
        self._scalers: dict[str, tuple[float, float]] = {}
        self._train_data: pd.DataFrame | None = None
        self._use_torch: bool = False

    @property
    def name(self) -> str:
        return "lstm"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "LSTMForecaster":
        self._train_data = Y_train
        try:
            import torch
            import torch.nn as nn

            self._use_torch = True

            class _LSTMModel(nn.Module):
                def __init__(self, hidden_size: int, num_layers: int):
                    super().__init__()
                    self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, 1)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values.astype(np.float32)
                if len(y) < self.lookback + 1:
                    continue
                # Normalize
                y_min, y_max = float(y.min()), float(y.max())
                scale = y_max - y_min if y_max != y_min else 1.0
                y_norm = (y - y_min) / scale
                self._scalers[str(uid)] = (y_min, scale)

                # Build sequences
                X_seq, Y_seq = [], []
                for i in range(len(y_norm) - self.lookback):
                    X_seq.append(y_norm[i : i + self.lookback])
                    Y_seq.append(y_norm[i + self.lookback])
                X_t = torch.tensor(np.array(X_seq), dtype=torch.float32).unsqueeze(-1)
                Y_t = torch.tensor(np.array(Y_seq), dtype=torch.float32).unsqueeze(-1)

                model = _LSTMModel(self.hidden_size, self.num_layers)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                loss_fn = nn.MSELoss()

                model.train()
                for _ in range(self.epochs):
                    pred = model(X_t)
                    loss = loss_fn(pred, Y_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                self._models[str(uid)] = model
            logger.info("model_fitted", model="lstm", n_series=len(self._models))

        except ImportError:
            logger.warning("torch_not_installed", fallback="weighted_rolling_avg")
            self._use_torch = False
            # Fallback: store exponentially-weighted statistics
            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values
                alpha = 2.0 / (min(self.lookback, len(y)) + 1)
                weights = np.array([(1 - alpha) ** i for i in range(len(y))])[::-1]
                weights /= weights.sum()
                ewm_mean = float(np.dot(weights, y))
                # Weighted trend from last lookback values
                tail = y[-min(self.lookback, len(y)) :]
                if len(tail) >= 2:
                    slope = float(np.polyfit(range(len(tail)), tail, 1)[0])
                else:
                    slope = 0.0
                self._models[str(uid)] = {"ewm_mean": ewm_mean, "slope": slope}
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if self._use_torch and str(uid) in self._models:
                import torch

                y = grp["y"].values.astype(np.float32)
                y_min, scale = self._scalers[str(uid)]
                y_norm = (y - y_min) / scale
                model = self._models[str(uid)]
                seq = y_norm[-self.lookback :].tolist()

                model.eval()
                with torch.no_grad():
                    for ds in future_dates:
                        x = torch.tensor([seq[-self.lookback :]], dtype=torch.float32).unsqueeze(-1)
                        pred_norm = model(x).item()
                        yhat = pred_norm * scale + y_min
                        records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})
                        seq.append(pred_norm)
            elif str(uid) in self._models:
                info = self._models[str(uid)]
                for step, ds in enumerate(future_dates):
                    yhat = info["ewm_mean"] + info["slope"] * (step + 1)
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 9. N-BEATS Forecaster
# ---------------------------------------------------------------------------


class NBEATSForecaster(ForecastModel):
    """N-BEATS neural architecture for time series forecasting.

    Falls back to polynomial trend + Fourier harmonics decomposition
    when ``neuralforecast`` is not installed.
    """

    def __init__(
        self,
        input_size: int = 30,
        h: int = 14,
        max_steps: int = 100,
        n_harmonics: int = 3,
        poly_degree: int = 2,
    ):
        self.input_size = input_size
        self._h = h
        self.max_steps = max_steps
        self.n_harmonics = n_harmonics
        self.poly_degree = poly_degree
        self._nf = None
        self._train_data: pd.DataFrame | None = None
        self._use_neuralforecast: bool = False
        self._models: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "nbeats"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "NBEATSForecaster":
        self._train_data = Y_train
        try:
            from neuralforecast import NeuralForecast  # type: ignore[import-untyped]
            from neuralforecast.models import NBEATS  # type: ignore[import-untyped]

            self._use_neuralforecast = True
            nbeats_model = NBEATS(
                input_size=self.input_size,
                h=self._h,
                max_steps=self.max_steps,
                scaler_type="standard",
            )
            nf = NeuralForecast(models=[nbeats_model], freq="D")
            nf.fit(df=Y_train)
            self._nf = nf
            logger.info("model_fitted", model="nbeats", backend="neuralforecast")
        except ImportError:
            logger.warning("neuralforecast_not_installed", fallback="poly_fourier")
            self._use_neuralforecast = False
            # Fallback: polynomial trend + Fourier harmonics
            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values.astype(np.float64)
                n = len(y)
                t = np.arange(n, dtype=np.float64)

                # Polynomial trend
                poly_coeffs = np.polyfit(t, y, min(self.poly_degree, max(1, n - 1)))

                # Fourier harmonics on residuals
                trend_vals = np.polyval(poly_coeffs, t)
                residuals = y - trend_vals
                harmonics = []
                period = min(7.0, float(n) / 2)
                for k in range(1, self.n_harmonics + 1):
                    cos_k = np.cos(2 * np.pi * k * t / period)
                    sin_k = np.sin(2 * np.pi * k * t / period)
                    a_k = 2.0 * np.dot(residuals, cos_k) / n
                    b_k = 2.0 * np.dot(residuals, sin_k) / n
                    harmonics.append((a_k, b_k, period / k))

                self._models[str(uid)] = {
                    "poly_coeffs": poly_coeffs.tolist(),
                    "harmonics": harmonics,
                    "n": n,
                    "period": period,
                }
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        if self._use_neuralforecast and self._nf is not None:
            forecasts = self._nf.predict()
            forecasts = forecasts.reset_index()
            # NeuralForecast column name is the model class name
            yhat_col = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
            if yhat_col:
                forecasts = forecasts.rename(columns={yhat_col[0]: "y_hat"})
            forecasts["y_hat"] = forecasts["y_hat"].clip(lower=0)
            return forecasts[["unique_id", "ds", "y_hat"]].head(
                h * self._train_data["unique_id"].nunique()
            )

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if str(uid) not in self._models:
                continue

            info = self._models[str(uid)]
            poly_coeffs = np.array(info["poly_coeffs"])
            n = info["n"]
            period = info["period"]

            for step, ds in enumerate(future_dates):
                t_val = float(n + step)
                yhat = float(np.polyval(poly_coeffs, t_val))
                # Use harmonic index k to match fit-time Fourier computation
                for k_idx, (a_k, b_k, _) in enumerate(info["harmonics"]):
                    k = k_idx + 1
                    yhat += a_k * np.cos(2 * np.pi * k * (n + step) / period)
                    yhat += b_k * np.sin(2 * np.pi * k * (n + step) / period)
                records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, float(yhat))})

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 10. Temporal Fusion Transformer (TFT)
# ---------------------------------------------------------------------------


class TFTForecaster(ForecastModel):
    """Temporal Fusion Transformer for multi-horizon forecasting.

    Falls back to sklearn feature-weighted linear regression when
    neural libraries are not installed.
    """

    def __init__(
        self,
        input_size: int = 30,
        h: int = 14,
        max_steps: int = 100,
    ):
        self.input_size = input_size
        self._h = h
        self.max_steps = max_steps
        self._nf = None
        self._train_data: pd.DataFrame | None = None
        self._use_neural: bool = False
        self._models: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "tft"

    def fit(self, Y_train: pd.DataFrame, X_train: pd.DataFrame | None = None) -> "TFTForecaster":
        self._train_data = Y_train
        try:
            from neuralforecast import NeuralForecast  # type: ignore[import-untyped]
            from neuralforecast.models import TFT  # type: ignore[import-untyped]

            self._use_neural = True
            tft_model = TFT(
                input_size=self.input_size,
                h=self._h,
                max_steps=self.max_steps,
                scaler_type="standard",
            )
            nf = NeuralForecast(models=[tft_model], freq="D")
            nf.fit(df=Y_train)
            self._nf = nf
            logger.info("model_fitted", model="tft", backend="neuralforecast")
        except ImportError:
            logger.warning("neuralforecast_not_installed", fallback="feature_weighted_regression")
            self._use_neural = False
            from sklearn.linear_model import Ridge

            for uid, grp in Y_train.groupby("unique_id"):
                grp = grp.sort_values("ds")
                y = grp["y"].values
                n = len(y)
                if n < 4:
                    self._models[str(uid)] = {"mean": float(np.mean(y))}
                    continue

                # Build features: trend, day_of_week, lag-based
                t = np.arange(n, dtype=np.float64)
                dow = np.array([d.dayofweek for d in grp["ds"]], dtype=np.float64)
                month = np.array([d.month for d in grp["ds"]], dtype=np.float64)
                # Proper lag computation: use NaN for unavailable values instead of np.roll
                lag1 = np.empty_like(y, dtype=np.float64)
                lag1[0] = np.nan
                lag1[1:] = y[:-1]

                lag7 = np.empty_like(y, dtype=np.float64)
                lag7[:7] = np.nan
                lag7[7:] = y[:-7]

                # Fill NaNs with series mean to avoid leaking future information
                y_mean = float(np.nanmean(y))
                lag1 = np.nan_to_num(lag1, nan=y_mean)
                lag7 = np.nan_to_num(lag7, nan=y_mean)

                X_feat = np.column_stack([t, dow, month, lag1, lag7])
                reg = Ridge(alpha=1.0)
                reg.fit(X_feat, y)
                self._models[str(uid)] = {
                    "model": reg,
                    "n": n,
                    "last_values": y[-max(7, 1) :].tolist(),
                }
            logger.info("model_fitted", model="tft", backend="sklearn_fallback", n_series=len(self._models))
        return self

    def predict(self, h: int, X_future: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._train_data is None:
            return pd.DataFrame(columns=["unique_id", "ds", "y_hat"])

        if self._use_neural and self._nf is not None:
            forecasts = self._nf.predict()
            forecasts = forecasts.reset_index()
            yhat_col = [c for c in forecasts.columns if c not in ("unique_id", "ds")]
            if yhat_col:
                forecasts = forecasts.rename(columns={yhat_col[0]: "y_hat"})
            forecasts["y_hat"] = forecasts["y_hat"].clip(lower=0)
            return forecasts[["unique_id", "ds", "y_hat"]].head(
                h * self._train_data["unique_id"].nunique()
            )

        records = []
        for uid, grp in self._train_data.groupby("unique_id"):
            grp = grp.sort_values("ds")
            last_ds = grp["ds"].iloc[-1]
            future_dates = pd.date_range(start=last_ds + pd.Timedelta(days=1), periods=h, freq="D")

            if str(uid) not in self._models:
                continue

            info = self._models[str(uid)]
            if "model" not in info:
                # Simple mean fallback
                for ds in future_dates:
                    records.append({"unique_id": uid, "ds": ds, "y_hat": max(0, info["mean"])})
                continue

            reg = info["model"]
            n = info["n"]
            history = info["last_values"][:]

            for step, ds in enumerate(future_dates):
                t_val = float(n + step)
                dow_val = float(ds.dayofweek)
                month_val = float(ds.month)
                lag1_val = history[-1] if history else 0.0
                lag7_val = history[-7] if len(history) >= 7 else history[0]
                features = np.array([[t_val, dow_val, month_val, lag1_val, lag7_val]])
                yhat = float(reg.predict(features)[0])
                yhat = max(0, yhat)
                records.append({"unique_id": uid, "ds": ds, "y_hat": yhat})
                history.append(yhat)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class ForecastModelFactory:
    """Factory for creating forecast models from configuration."""

    _registry: dict[str, type[ForecastModel]] = {
        "naive_ma30": NaiveMovingAverage,
        "sarimax": SARIMAXForecaster,
        "xgboost": XGBoostForecaster,
        "lightgbm": LightGBMForecaster,
        "chronos2_zs": ChronosForecaster,
        "routing_ensemble": RoutingEnsemble,
        "prophet": ProphetForecaster,
        "lstm": LSTMForecaster,
        "nbeats": NBEATSForecaster,
        "tft": TFTForecaster,
    }

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> ForecastModel:
        """Create a forecast model by name.

        Args:
            name: Model name key.
            **kwargs: Model-specific parameters.

        Returns:
            Initialized ForecastModel instance.
        """
        model_cls = cls._registry.get(name)
        if model_cls is None:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._registry.keys())}")
        return model_cls(**kwargs)

    @classmethod
    def create_all(cls) -> dict[str, ForecastModel]:
        """Create all registered models with default configs."""
        config = get_model_config()
        models = {}

        models["naive_ma30"] = NaiveMovingAverage(window=config.get("naive", {}).get("window", 30))

        sarimax_cfg = config.get("sarimax", {})
        models["sarimax"] = SARIMAXForecaster(
            order=tuple(sarimax_cfg.get("order", [1, 1, 1])),
            seasonal_order=tuple(sarimax_cfg.get("seasonal_order", [1, 1, 1, 7])),
        )

        xgb_cfg = config.get("xgboost", {})
        models["xgboost"] = XGBoostForecaster(
            n_estimators=xgb_cfg.get("n_estimators", 200),
            max_depth=xgb_cfg.get("max_depth", 5),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
        )

        lgb_cfg = config.get("lightgbm", {})
        models["lightgbm"] = LightGBMForecaster(
            n_estimators=lgb_cfg.get("n_estimators", 300),
            num_leaves=lgb_cfg.get("num_leaves", 31),
            learning_rate=lgb_cfg.get("learning_rate", 0.1),
        )

        chronos_cfg = config.get("chronos", {})
        models["chronos2_zs"] = ChronosForecaster(
            model_name=chronos_cfg.get("model_name", "amazon/chronos-t5-small"),
            prediction_length=chronos_cfg.get("prediction_length", 14),
        )

        routing_cfg = config.get("routing", {})
        models["routing_ensemble"] = RoutingEnsemble(
            cold_start_threshold_days=routing_cfg.get("cold_start_threshold_days", 60),
            intermittency_threshold=routing_cfg.get("intermittency_threshold", 0.5),
        )

        prophet_cfg = config.get("prophet", {})
        models["prophet"] = ProphetForecaster(
            yearly_seasonality=prophet_cfg.get("yearly_seasonality", True),
            weekly_seasonality=prophet_cfg.get("weekly_seasonality", True),
            changepoint_prior_scale=prophet_cfg.get("changepoint_prior_scale", 0.05),
        )

        lstm_cfg = config.get("lstm", {})
        models["lstm"] = LSTMForecaster(
            hidden_size=lstm_cfg.get("hidden_size", 32),
            num_layers=lstm_cfg.get("num_layers", 1),
            lookback=lstm_cfg.get("lookback", 30),
            epochs=lstm_cfg.get("epochs", 50),
            learning_rate=lstm_cfg.get("learning_rate", 0.001),
        )

        nbeats_cfg = config.get("nbeats", {})
        models["nbeats"] = NBEATSForecaster(
            input_size=nbeats_cfg.get("input_size", 30),
            h=nbeats_cfg.get("horizon", 14),
            max_steps=nbeats_cfg.get("max_steps", 100),
            n_harmonics=nbeats_cfg.get("n_harmonics", 3),
            poly_degree=nbeats_cfg.get("poly_degree", 2),
        )

        tft_cfg = config.get("tft", {})
        models["tft"] = TFTForecaster(
            input_size=tft_cfg.get("input_size", 30),
            h=tft_cfg.get("horizon", 14),
            max_steps=tft_cfg.get("max_steps", 100),
        )

        return models

    @classmethod
    def available_models(cls) -> list[str]:
        """List all registered model names."""
        return list(cls._registry.keys())
