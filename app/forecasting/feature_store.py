"""Feature Store pattern for training-serving consistency.

Eliminates training-serving skew by maintaining a single source of truth
for feature computation.

Two modes:
- Offline store: batch ETL (daily) → used for training
- Online store: real-time query → used for API serving

Consistency model: Eventual Consistency (AP > CP in CAP theorem)
Rationale: Forecasting tolerates 1-day feature lag; availability matters more.

Feature groups:
- demand_features: lag-based features computed from Y_df
- exogenous_features: promo, holiday, temperature
- derived_features: rolling stats, intermittency ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from app.log_config import get_logger

logger = get_logger(__name__)


@dataclass
class Feature:
    """Single feature definition."""

    name: str
    dtype: str
    description: str = ""


@dataclass
class FeatureGroup:
    """Group of related features with metadata."""

    name: str
    entities: list[str]
    features: list[Feature]
    ttl: timedelta = field(default_factory=lambda: timedelta(days=1))
    description: str = ""


# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------

DEMAND_FEATURES = FeatureGroup(
    name="demand_features",
    entities=["unique_id"],
    features=[
        Feature("lag_1", "float64", "1-day lagged demand"),
        Feature("lag_7", "float64", "7-day lagged demand"),
        Feature("lag_14", "float64", "14-day lagged demand"),
        Feature("lag_28", "float64", "28-day lagged demand"),
        Feature("rolling_mean_7", "float64", "7-day rolling mean"),
        Feature("rolling_std_7", "float64", "7-day rolling std"),
        Feature("rolling_mean_28", "float64", "28-day rolling mean"),
        Feature("day_of_week", "int64", "Day of week (0=Mon)"),
        Feature("month", "int64", "Month (1-12)"),
        Feature("is_holiday", "int64", "Holiday flag"),
        Feature("promo_flag", "int64", "Promotion flag"),
        Feature("day_of_year", "int64", "Day of year (1-366)"),
    ],
    ttl=timedelta(days=1),
    description="Lag-based demand features for forecasting models",
)

EXOGENOUS_FEATURES = FeatureGroup(
    name="exogenous_features",
    entities=["unique_id"],
    features=[
        Feature("temperature", "float64", "Local temperature"),
        Feature("price", "float64", "Current unit price"),
        Feature("stock_level", "float64", "Current stock level"),
    ],
    ttl=timedelta(days=1),
    description="External features not derived from demand",
)


ALL_FEATURE_GROUPS = [DEMAND_FEATURES, EXOGENOUS_FEATURES]


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------


class FeatureStore:
    """Dual-mode feature store (offline + online).

    Offline: batch computation from historical data → for model training.
    Online: point-in-time lookup → for real-time API serving.

    Consistency: Eventual Consistency (AP > CP).
    - Offline store updates daily via batch ETL
    - Online store may lag by up to 1 day (TTL-based refresh)
    - This is acceptable for forecasting (not real-time trading)
    """

    def __init__(self) -> None:
        self._offline_store: pd.DataFrame | None = None
        self._online_store: dict[str, dict[str, Any]] = {}
        self._last_update: pd.Timestamp | None = None

    def materialize_offline(
        self,
        Y_df: pd.DataFrame,
        X_future: pd.DataFrame | None = None,
        X_past: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute all features from raw data (batch mode).

        This is the single source of truth for feature computation.
        Both training and serving use features computed by this method.

        Args:
            Y_df: Demand data (unique_id, ds, y).
            X_future: Future exogenous features.
            X_past: Historical exogenous features.

        Returns:
            Feature DataFrame with all computed features.
        """
        features = self._compute_demand_features(Y_df)

        if X_future is not None:
            features = features.merge(
                X_future[["unique_id", "ds", "promo_flag", "is_holiday", "temperature"]],
                on=["unique_id", "ds"],
                how="left",
            )

        if X_past is not None:
            features = features.merge(
                X_past[["unique_id", "ds", "price", "stock_level"]],
                on=["unique_id", "ds"],
                how="left",
            )

        self._offline_store = features
        self._last_update = pd.Timestamp.now()
        logger.info(
            "offline_store_materialized",
            n_rows=len(features),
            n_features=len(features.columns),
        )

        return features

    def get_training_features(
        self,
        unique_ids: list[str] | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Get features for model training (offline mode).

        Args:
            unique_ids: Filter to specific SKUs.
            start_date: Start of training window.
            end_date: End of training window.

        Returns:
            Filtered feature DataFrame.
        """
        if self._offline_store is None:
            raise ValueError("Offline store not materialized. Call materialize_offline() first.")

        df = self._offline_store.copy()

        if unique_ids is not None:
            df = df[df["unique_id"].isin(unique_ids)]
        if start_date is not None:
            df = df[df["ds"] >= start_date]
        if end_date is not None:
            df = df[df["ds"] <= end_date]

        return df

    def update_online(self, unique_id: str, features: dict[str, Any]) -> None:
        """Update online store for a single entity (real-time mode).

        Args:
            unique_id: SKU identifier.
            features: Dict of feature_name -> value.
        """
        self._online_store[unique_id] = {
            **features,
            "_updated_at": pd.Timestamp.now(),
        }

    def get_online_features(self, unique_id: str) -> dict[str, Any] | None:
        """Get features for a single entity (real-time serving).

        Args:
            unique_id: SKU identifier.

        Returns:
            Feature dict or None if not found.
        """
        return self._online_store.get(unique_id)

    def _compute_demand_features(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        """Compute lag and rolling features from demand data."""
        Y_sorted = Y_df.sort_values(["unique_id", "ds"]).copy()

        # Lag features (vectorized groupby)
        for lag in [1, 7, 14, 28]:
            Y_sorted[f"lag_{lag}"] = Y_sorted.groupby("unique_id")["y"].shift(lag)

        # Rolling features (shifted by 1 to prevent leakage)
        shifted = Y_sorted.groupby("unique_id")["y"].shift(1)
        Y_sorted["rolling_mean_7"] = shifted.groupby(Y_sorted["unique_id"]).rolling(7, min_periods=1).mean().droplevel(0)
        Y_sorted["rolling_std_7"] = shifted.groupby(Y_sorted["unique_id"]).rolling(7, min_periods=1).std().droplevel(0).fillna(0)
        Y_sorted["rolling_mean_28"] = shifted.groupby(Y_sorted["unique_id"]).rolling(28, min_periods=1).mean().droplevel(0)

        # Calendar features
        Y_sorted["day_of_week"] = Y_sorted["ds"].dt.dayofweek
        Y_sorted["month"] = Y_sorted["ds"].dt.month
        Y_sorted["day_of_year"] = Y_sorted["ds"].dt.dayofyear

        return Y_sorted.reset_index(drop=True)

    @property
    def last_update(self) -> pd.Timestamp | None:
        """Timestamp of last offline store materialization."""
        return self._last_update

    @property
    def feature_names(self) -> list[str]:
        """List all feature column names."""
        if self._offline_store is None:
            return []
        exclude = {"unique_id", "ds", "y"}
        return [c for c in self._offline_store.columns if c not in exclude]
