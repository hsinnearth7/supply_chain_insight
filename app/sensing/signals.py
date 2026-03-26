"""Demand signal processing and near-term forecast adjustment.

Ingests multiple signal sources (POS, social, weather) and adjusts
base forecasts for short-horizon accuracy improvement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.settings import get_sensing_config

logger = logging.getLogger(__name__)


@dataclass
class DemandSignal:
    """A demand signal from an external source."""
    source: str
    timestamp: str
    product_id: str
    signal_value: float
    confidence: float = 1.0


@dataclass
class Spike:
    """Detected demand spike."""
    product_id: str
    period: str
    magnitude: float
    sigma: float
    source: str


class SignalProcessor:
    """Processes demand signals and adjusts forecasts."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or get_sensing_config()
        self.pos_weight = self.config.get("pos_weight", 0.6)
        self.social_weight = self.config.get("social_weight", 0.25)
        self.weather_weight = self.config.get("weather_weight", 0.15)
        self.decay_half_life = self.config.get("decay_half_life_days", 7)
        self.spike_threshold = self.config.get("spike_threshold_sigma", 2.5)
        self.horizon = self.config.get("sensing_horizon_days", 14)

    def generate_synthetic_signals(
        self,
        product_ids: list[str],
        n_days: int = 14,
        seed: int = 42,
    ) -> list[DemandSignal]:
        """Generate synthetic demand signals for demo/testing."""
        rng = np.random.RandomState(seed)
        signals = []
        sources = ["pos", "social", "weather"]

        for day in range(n_days):
            for pid in product_ids[:10]:  # limit to 10 products
                source = sources[day % len(sources)]
                base_value = rng.normal(100, 20)
                # Add occasional spike
                if rng.random() < 0.05:
                    base_value *= rng.uniform(1.5, 3.0)

                signals.append(DemandSignal(
                    source=source,
                    timestamp=f"day_{day}",
                    product_id=pid,
                    signal_value=float(max(0, base_value)),
                    confidence=float(rng.uniform(0.6, 1.0)),
                ))

        return signals

    def compute_signal_weight(self, signal: DemandSignal, horizon_days: int) -> float:
        """Calculate weight with exponential decay based on recency."""
        source_weights = {
            "pos": self.pos_weight,
            "social": self.social_weight,
            "weather": self.weather_weight,
        }
        base_weight = source_weights.get(signal.source, 0.3)

        # Recency decay: extract day number from timestamp (e.g. "day_3")
        try:
            day_num = int(signal.timestamp.split("_")[-1])
        except (ValueError, IndexError):
            day_num = 0
        # Exponential decay: newer signals (higher day_num) get more weight
        max_day = horizon_days - 1 if horizon_days > 1 else 1
        days_ago = max(0, max_day - day_num)
        decay = 0.5 ** (days_ago / max(1, self.decay_half_life))

        # Shorter horizon = signals matter more
        horizon_factor = max(0.1, 1.0 - (horizon_days / (self.horizon * 2)))

        return base_weight * signal.confidence * horizon_factor * decay

    def compute_sensing_adjustment(
        self,
        base_forecast: pd.DataFrame,
        signals: list[DemandSignal],
    ) -> pd.DataFrame:
        """Adjust base forecast using demand signals.

        Args:
            base_forecast: DataFrame with columns [product_id, period, forecast].
            signals: List of demand signals.

        Returns:
            DataFrame with additional 'adjusted_forecast' and 'adjustment_pct' columns.
        """
        df = base_forecast.copy()
        df["adjusted_forecast"] = df["forecast"]
        df["adjustment_pct"] = 0.0

        if not signals:
            return df

        # Group signals by product
        signal_by_product: dict[str, list[DemandSignal]] = {}
        for s in signals:
            signal_by_product.setdefault(s.product_id, []).append(s)

        for idx, row in df.iterrows():
            pid = row.get("product_id", "")
            product_signals = signal_by_product.get(pid, [])
            if not product_signals:
                continue

            # Weighted average of signals
            weights = [self.compute_signal_weight(s, self.horizon) for s in product_signals]
            total_weight = sum(weights)
            if total_weight == 0:
                continue

            weighted_signal = sum(
                s.signal_value * w for s, w in zip(product_signals, weights)
            ) / total_weight

            # Blend: 70% base forecast + 30% signal
            blend_factor = 0.3
            base_val = float(row["forecast"])
            adjusted = base_val * (1 - blend_factor) + weighted_signal * blend_factor

            df.at[idx, "adjusted_forecast"] = adjusted
            if base_val > 0:
                df.at[idx, "adjustment_pct"] = ((adjusted - base_val) / base_val) * 100

        return df

    def detect_spikes(self, signals: list[DemandSignal]) -> list[Spike]:
        """Detect unusual demand spikes from signals."""
        if not signals:
            return []

        # Group signals (full objects) by product
        by_product: dict[str, list[DemandSignal]] = {}
        for s in signals:
            by_product.setdefault(s.product_id, []).append(s)

        spikes = []
        for pid, product_signals in by_product.items():
            values = [s.signal_value for s in product_signals]
            arr = np.array(values)
            if len(arr) < 2:
                continue  # Need at least 2 observations for sample std
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1))

            for i, (v, sig) in enumerate(zip(values, product_signals)):
                sigma = (v - mean) / std if std > 0 else 0
                if sigma > self.spike_threshold:
                    spikes.append(Spike(
                        product_id=pid,
                        period=f"signal_{i}",
                        magnitude=float(v),
                        sigma=float(sigma),
                        source=sig.source,
                    ))

        return spikes
