"""Hierarchical forecasting with reconciliation.

4-layer hierarchy: National (1) → Warehouse (3) → Category (60) → SKU (200)
Total: 224 nodes (adjustable based on HierarchySpec)

Reconciliation methods:
- BottomUp: aggregate SKU forecasts upward (baseline)
- MinTrace(ols): optimal linear reconciliation minimizing trace of covariance
- ERM: empirical risk minimization

Target: MinTrace MAPE 8% lower than BottomUp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.forecasting.data_generator import HierarchySpec, build_hierarchy_matrix
from app.log_config import get_logger

logger = get_logger(__name__)


def aggregate_to_hierarchy(
    Y_df: pd.DataFrame,
    S_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    """Aggregate bottom-level time series to all hierarchy levels.

    Args:
        Y_df: Bottom-level demand (unique_id, ds, y).
        S_df: Static attributes with hierarchy columns.

    Returns:
        Y_agg: Aggregated time series for all levels.
        S: Summation matrix.
        tags: Level-to-node mapping.
    """
    S, tags = build_hierarchy_matrix(S_df)

    # Merge hierarchy info
    Y_with_hier = Y_df.merge(S_df, on="unique_id", how="left")

    agg_records = []

    # Level 0: National
    national = Y_with_hier.groupby("ds")["y"].sum().reset_index()
    national["unique_id"] = "Total"
    agg_records.append(national)

    # Level 1: Warehouse
    for wh in tags["warehouse"]:
        wh_data = Y_with_hier[Y_with_hier["warehouse"] == wh].groupby("ds")["y"].sum().reset_index()
        wh_data["unique_id"] = wh
        agg_records.append(wh_data)

    # Level 2: Warehouse/Category/Subcategory
    for cat_key in tags["category"]:
        parts = cat_key.split("/")
        wh, cat, subcat = parts[0], parts[1], parts[2]
        mask = (Y_with_hier["warehouse"] == wh) & (Y_with_hier["category"] == cat) & (Y_with_hier["subcategory"] == subcat)
        cat_data = Y_with_hier[mask].groupby("ds")["y"].sum().reset_index()
        cat_data["unique_id"] = cat_key
        agg_records.append(cat_data)

    # Level 3: SKU (already in Y_df)
    for uid in tags["sku"]:
        sku_data = Y_df[Y_df["unique_id"] == uid][["unique_id", "ds", "y"]].copy()
        agg_records.append(sku_data)

    Y_agg = pd.concat(agg_records, ignore_index=True)
    logger.info(
        "hierarchy_aggregated",
        n_series=Y_agg["unique_id"].nunique(),
        n_dates=Y_agg["ds"].nunique(),
        total_rows=len(Y_agg),
    )

    return Y_agg, S, tags


class HierarchicalForecaster:
    """Hierarchical forecast reconciliation.

    Supports BottomUp and MinTrace reconciliation methods.
    Uses hierarchicalforecast library when available,
    falls back to manual implementation.
    """

    def __init__(self, method: str = "mint_ols"):
        """Initialize reconciler.

        Args:
            method: Reconciliation method. One of:
                - "bottom_up": Simple bottom-up aggregation
                - "mint_ols": MinTrace with OLS covariance estimation
                - "mint_wls": MinTrace with WLS (diagonal covariance)
        """
        self.method = method
        self._S: np.ndarray | None = None
        self._tags: dict[str, list[str]] | None = None

    def reconcile(
        self,
        Y_hat: pd.DataFrame,
        S: np.ndarray,
        tags: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Reconcile base forecasts to ensure hierarchical consistency.

        Args:
            Y_hat: Base forecasts for all hierarchy levels (unique_id, ds, y_hat).
            S: Summation matrix (n_total × n_bottom).
            tags: Level-to-node mapping.

        Returns:
            Reconciled forecasts (unique_id, ds, y_hat_reconciled).
        """
        self._S = S
        self._tags = tags

        try:
            return self._reconcile_with_library(Y_hat, S, tags)
        except ImportError:
            logger.warning("hierarchicalforecast_not_installed", fallback="manual_reconciliation")
            return self._reconcile_manual(Y_hat, S, tags)

    def _reconcile_with_library(
        self,
        Y_hat: pd.DataFrame,
        S: np.ndarray,
        tags: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Reconcile using hierarchicalforecast library."""
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.methods import BottomUp, MinTrace

        if self.method == "bottom_up":
            reconcilers = [BottomUp()]
        elif self.method == "mint_ols":
            reconcilers = [BottomUp(), MinTrace(method="ols")]
        elif self.method == "mint_wls":
            reconcilers = [BottomUp(), MinTrace(method="wls_var")]
        else:
            reconcilers = [MinTrace(method="ols")]

        hrec = HierarchicalReconciliation(reconcilers=reconcilers)

        # The library expects specific column naming
        Y_hat_formatted = Y_hat.rename(columns={"y_hat": "base_forecast"})

        try:
            Y_reconciled = hrec.reconcile(
                Y_hat_df=Y_hat_formatted,
                S=S,
                tags=tags,
            )
            logger.info("reconciliation_complete", method=self.method)
            return Y_reconciled
        except Exception as e:
            logger.warning("library_reconciliation_failed", error=str(e), fallback="manual")
            return self._reconcile_manual(Y_hat, S, tags)

    def _reconcile_manual(
        self,
        Y_hat: pd.DataFrame,
        S: np.ndarray,
        tags: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Manual reconciliation implementation (vectorized).

        Instead of looping per date, pivot to matrix, multiply, unpivot.

        BottomUp: Y_reconciled = S × Y_hat_bottom
        MinTrace(OLS): Y_reconciled = S × (S'S)^{-1} × S' × Y_hat_all
        """
        # INVARIANT: tags dict must preserve insertion order matching the summation
        # matrix S rows: [national, warehouse, category, sku]. Python 3.7+ dicts
        # maintain insertion order. Do NOT reorder tags keys.
        all_ids = []
        for level_ids in tags.values():
            all_ids.extend(level_ids)

        bottom_ids = tags["sku"]
        n_bottom = len(bottom_ids)

        # Pivot to matrix: rows=dates, columns=unique_ids
        pivot_df = Y_hat.pivot(index="ds", columns="unique_id", values="y_hat")

        if self.method == "bottom_up":
            # Reorder columns to match bottom-level order, fill missing with 0
            bottom_pivot = pivot_df.reindex(columns=bottom_ids, fill_value=0)
            # Matrix multiply: (n_dates, n_bottom) @ S.T -> (n_dates, n_all)
            # S shape is (n_all, n_bottom), so bottom_values @ S.T gives (n_dates, n_all)
            reconciled_matrix = bottom_pivot.values @ S.T
        else:  # MinTrace OLS
            # NOTE: This implements OLS reconciliation (W=I), equivalent to projection onto col(S).
            # For true MinTrace, W should be the forecast error covariance matrix.
            # Reorder columns to match all_ids order, fill missing with 0
            all_pivot = pivot_df.reindex(columns=all_ids, fill_value=0)
            # MinTrace: P = S(S'S)^{-1}S'
            StS = S.T @ S
            try:
                StS_inv = np.linalg.inv(StS)
                P = S @ StS_inv @ S.T
                # (n_dates, n_all) @ P.T -> (n_dates, n_all)
                reconciled_matrix = all_pivot.values @ P.T
            except np.linalg.LinAlgError:
                # Fallback to bottom-up if singular
                bottom_pivot = pivot_df.reindex(columns=bottom_ids, fill_value=0)
                reconciled_matrix = bottom_pivot.values @ S.T

        # Ensure non-negative
        reconciled_matrix = np.maximum(reconciled_matrix, 0)

        # Build result DataFrame from matrix
        dates = pivot_df.index
        result = pd.DataFrame(reconciled_matrix, index=dates, columns=all_ids)
        result = result.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y_hat_reconciled")

        logger.info("manual_reconciliation_complete", method=self.method, n_records=len(result))
        return result

    def evaluate_reconciliation(
        self,
        Y_actual: pd.DataFrame,
        Y_bottom_up: pd.DataFrame,
        Y_mint: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compare BottomUp vs MinTrace MAPE.

        Returns:
            Dict with comparison metrics.
        """
        from app.forecasting.evaluation import mape as compute_mape

        # Merge actuals with forecasts
        merged_bu = Y_actual.merge(Y_bottom_up, on=["unique_id", "ds"], how="inner")
        merged_mt = Y_actual.merge(Y_mint, on=["unique_id", "ds"], how="inner")

        mape_bu = compute_mape(merged_bu["y"].values, merged_bu["y_hat_reconciled"].values)
        mape_mt = compute_mape(merged_mt["y"].values, merged_mt["y_hat_reconciled"].values)

        improvement = ((mape_bu - mape_mt) / mape_bu) * 100 if mape_bu > 0 else 0

        return {
            "mape_bottom_up": round(mape_bu, 2),
            "mape_mintrace": round(mape_mt, 2),
            "improvement_pct": round(improvement, 1),
            "target_improvement": 8.0,
            "target_met": improvement >= 8.0,
        }
