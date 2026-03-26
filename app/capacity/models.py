"""Capacity planning models and analysis.

Compares aggregated demand forecasts against production capacity
to identify bottlenecks and suggest adjustments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.settings import get_capacity_config

logger = logging.getLogger(__name__)


@dataclass
class CapacityProfile:
    """Production line capacity profile."""
    line_id: str
    product_group: str
    max_throughput_per_day: float
    changeover_time_hours: float = 2.0
    efficiency_factor: float = 0.85
    maintenance_window_hours: float = 4.0  # per week


@dataclass
class Bottleneck:
    """Identified capacity bottleneck."""
    period: str
    line_id: str
    demand: float
    capacity: float
    deficit: float
    utilization: float


@dataclass
class Adjustment:
    """Suggested capacity adjustment."""
    bottleneck: Bottleneck
    action: str  # "overtime", "outsource", "demand_smoothing"
    additional_capacity: float
    estimated_cost: float


@dataclass
class FeasibilityReport:
    """Capacity feasibility analysis result."""
    feasible: bool
    avg_utilization: float
    max_utilization: float
    bottleneck_count: int
    bottlenecks: list[Bottleneck] = field(default_factory=list)
    period_utilizations: dict[str, float] = field(default_factory=dict)
    demand_coverage: float = 1.0


class CapacityPlanner:
    """Analyzes production capacity against demand forecasts."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or get_capacity_config()
        self.production_lines = self.config.get("production_lines", 3)
        self.shift_hours = self.config.get("shift_hours", 8)
        self.max_overtime = self.config.get("max_overtime_hours", 4)
        self.changeover_cost = self.config.get("changeover_cost_per_hour", 150)
        self.utilization_target = self.config.get("utilization_target", 0.85)
        self.horizon_days = self.config.get("planning_horizon_days", 90)

    def build_default_profiles(self) -> list[CapacityProfile]:
        """Generate default capacity profiles from config."""
        profiles = []
        for i in range(self.production_lines):
            profiles.append(CapacityProfile(
                line_id=f"LINE-{i+1:02d}",
                product_group=f"group_{chr(65 + i)}",
                max_throughput_per_day=1000.0 * (1.0 + i * 0.2),
                changeover_time_hours=2.0,
                efficiency_factor=self.utilization_target,
            ))
        return profiles

    def check_feasibility(
        self,
        demand_forecast: pd.DataFrame,
        profiles: list[CapacityProfile] | None = None,
    ) -> FeasibilityReport:
        """Check if demand can be met by available capacity.

        Args:
            demand_forecast: DataFrame with columns [period, demand].
            profiles: Capacity profiles. Uses defaults if None.

        Returns:
            FeasibilityReport with utilization and bottleneck details.
        """
        profiles = profiles or self.build_default_profiles()
        total_daily_capacity = sum(
            p.max_throughput_per_day * p.efficiency_factor for p in profiles
        )

        if demand_forecast.empty:
            return FeasibilityReport(
                feasible=True, avg_utilization=0.0, max_utilization=0.0,
                bottleneck_count=0, demand_coverage=1.0,
            )

        # Group by period (weekly buckets)
        df = demand_forecast.copy()
        if "period" not in df.columns:
            df["period"] = [f"W{i//7 + 1}" for i in range(len(df))]

        period_demand = df.groupby("period")["demand"].sum()
        days_per_period = 7  # Weekly periods from orchestrator
        period_capacity = total_daily_capacity * days_per_period

        bottlenecks = []
        utilizations = {}

        for period, demand in period_demand.items():
            util = demand / period_capacity if period_capacity > 0 else float("inf")
            utilizations[str(period)] = min(util, 2.0)

            if util > 1.0:
                bottlenecks.append(Bottleneck(
                    period=str(period),
                    line_id="ALL",
                    demand=float(demand),
                    capacity=float(period_capacity),
                    deficit=float(demand - period_capacity),
                    utilization=float(util),
                ))

        utils_array = np.array(list(utilizations.values()))
        avg_util = float(np.mean(utils_array)) if len(utils_array) > 0 else 0.0
        max_util = float(np.max(utils_array)) if len(utils_array) > 0 else 0.0
        total_demand = float(period_demand.sum())
        total_cap = period_capacity * len(period_demand)
        coverage = min(1.0, total_cap / total_demand) if total_demand > 0 else 1.0

        return FeasibilityReport(
            feasible=len(bottlenecks) == 0,
            avg_utilization=avg_util,
            max_utilization=max_util,
            bottleneck_count=len(bottlenecks),
            bottlenecks=bottlenecks,
            period_utilizations=utilizations,
            demand_coverage=coverage,
        )

    def suggest_adjustments(self, bottlenecks: list[Bottleneck]) -> list[Adjustment]:
        """Suggest capacity adjustments for identified bottlenecks."""
        adjustments = []
        overtime_capacity_per_day = sum(
            self.max_overtime * (1000.0 / self.shift_hours)
            for _ in range(self.production_lines)
        )

        for bn in bottlenecks:
            if bn.deficit <= overtime_capacity_per_day * 7:
                adjustments.append(Adjustment(
                    bottleneck=bn,
                    action="overtime",
                    additional_capacity=overtime_capacity_per_day * 7,
                    estimated_cost=self.max_overtime * self.changeover_cost * 7,
                ))
            else:
                adjustments.append(Adjustment(
                    bottleneck=bn,
                    action="outsource",
                    additional_capacity=bn.deficit,
                    estimated_cost=bn.deficit * 2.0,
                ))

        return adjustments
