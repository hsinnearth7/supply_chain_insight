"""Synthetic demand data generator in Nixtla long format.

Generates realistic retail demand data with M5-style statistical properties:
1. Intermittent demand (30% of SKUs have 50%+ zero-demand days)
2. Long-tail distribution (Negative Binomial, not Normal)
3. Price elasticity (price +10% → demand -5% to -15%, category-dependent)
4. Substitution effects (cross-elasticity between same-category SKUs)
5. Censored demand (stock=0 → observed demand=0, true demand>0)

Output format follows Nixtla conventions:
- Y_df:      (unique_id, ds, y)              — demand time series
- S_df:      (unique_id, ...)                — static attributes
- X_future:  (unique_id, ds, ...)            — known future exogenous
- X_past:    (unique_id, ds, ...)            — historical dynamic exogenous

Hierarchy: National (1) → Warehouse (3) → Category (20) → SKU (200)
Summation matrix S: 224 × 200
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa

from app.seed import set_global_seed

# ---------------------------------------------------------------------------
# Pandera Schemas (Data Contracts)
# ---------------------------------------------------------------------------

Y_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str, nullable=False),
        "ds": pa.Column("datetime64[ns]", nullable=False),
        "y": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    strict=True,
    coerce=True,
)

S_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str, nullable=False),
        "warehouse": pa.Column(str, nullable=False),
        "category": pa.Column(str, nullable=False),
        "subcategory": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=True,
)

X_FUTURE_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str, nullable=False),
        "ds": pa.Column("datetime64[ns]", nullable=False),
        "promo_flag": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "is_holiday": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        "temperature": pa.Column(float, nullable=False),
    },
    strict=True,
    coerce=True,
)

X_PAST_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str, nullable=False),
        "ds": pa.Column("datetime64[ns]", nullable=False),
        "price": pa.Column(float, pa.Check.gt(0), nullable=False),
        "stock_level": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    strict=True,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Category / SKU metadata
# ---------------------------------------------------------------------------

CATEGORIES = {
    "Electronics": ["Phones", "Laptops", "Tablets", "Accessories", "Cameras"],
    "Grocery": ["Dairy", "Snacks", "Beverages", "Frozen", "Produce"],
    "Apparel": ["Mens", "Womens", "Kids", "Shoes", "Sportswear"],
    "Home": ["Furniture", "Kitchen", "Decor", "Bedding", "Garden"],
}

WAREHOUSES = ["NYC", "LAX", "CHI"]

# Category-specific demand parameters
CATEGORY_PARAMS: dict[str, dict[str, Any]] = {
    "Electronics": {
        "base_demand_mean": 15,
        "base_demand_var": 8,
        "price_range": (50.0, 500.0),
        "elasticity": -0.12,
        "seasonality_amp": 0.3,
    },
    "Grocery": {
        "base_demand_mean": 80,
        "base_demand_var": 30,
        "price_range": (2.0, 20.0),
        "elasticity": -0.08,
        "seasonality_amp": 0.15,
    },
    "Apparel": {
        "base_demand_mean": 25,
        "base_demand_var": 12,
        "price_range": (15.0, 150.0),
        "elasticity": -0.10,
        "seasonality_amp": 0.4,
    },
    "Home": {
        "base_demand_mean": 10,
        "base_demand_var": 5,
        "price_range": (20.0, 300.0),
        "elasticity": -0.06,
        "seasonality_amp": 0.2,
    },
}


@dataclass
class HierarchySpec:
    """4-layer hierarchy specification."""

    warehouses: list[str] = field(default_factory=lambda: WAREHOUSES)
    categories: dict[str, list[str]] = field(default_factory=lambda: CATEGORIES)
    skus_per_subcategory_per_warehouse: int = 2

    @property
    def n_skus(self) -> int:
        n_subcats = sum(len(v) for v in self.categories.values())
        return len(self.warehouses) * n_subcats * self.skus_per_subcategory_per_warehouse

    @property
    def n_nodes(self) -> int:
        """Total hierarchy nodes: 1 (national) + 3 (warehouse) + 20 (cat) + 200 (sku)."""
        n_subcats = sum(len(v) for v in self.categories.values())
        n_warehouse = len(self.warehouses)
        n_cat_level = n_warehouse * n_subcats
        return 1 + n_warehouse + n_cat_level + self.n_skus


def generate_sku_ids(spec: HierarchySpec) -> list[dict[str, str]]:
    """Generate SKU metadata with hierarchy attributes."""
    skus = []
    sku_counter = 0
    for warehouse in spec.warehouses:
        for category, subcategories in spec.categories.items():
            for subcategory in subcategories:
                for i in range(spec.skus_per_subcategory_per_warehouse):
                    sku_counter += 1
                    skus.append(
                        {
                            "unique_id": f"SKU_{sku_counter:04d}",
                            "warehouse": warehouse,
                            "category": category,
                            "subcategory": subcategory,
                        }
                    )
    return skus


def _generate_holidays(dates: pd.DatetimeIndex) -> np.ndarray:
    """Generate holiday flags for US retail calendar."""
    holidays = np.zeros(len(dates), dtype=int)
    for i, d in enumerate(dates):
        # Major US retail holidays (simplified)
        # TODO: Thanksgiving is the 4th Thursday of November, not always Nov 22-28.
        # Should use `d.weekday() == 3` (Thursday) and count occurrences.
        if (d.month == 11 and d.day >= 22 and d.day <= 28):  # Thanksgiving week
            holidays[i] = 1
        elif (d.month == 12 and d.day >= 15):  # Holiday season
            holidays[i] = 1
        elif (d.month == 7 and d.day == 4):  # July 4th
            holidays[i] = 1
        elif (d.month == 1 and d.day <= 3):  # New Year
            holidays[i] = 1
    return holidays


def _generate_temperature(dates: pd.DatetimeIndex, warehouse: str, rng: np.random.Generator) -> np.ndarray:
    """Generate temperature based on warehouse location and season."""
    base_temps = {"NYC": 55, "LAX": 70, "CHI": 50}
    base = base_temps.get(warehouse, 60)
    day_of_year = dates.dayofyear.values
    # Sinusoidal seasonal pattern
    seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = rng.normal(0, 3, len(dates))
    return base + seasonal + noise


def generate_demand_data(
    seed: int = 42,
    history_days: int = 730,
    start_date: str = "2022-01-01",
    spec: HierarchySpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate complete demand dataset in Nixtla long format.

    Args:
        seed: Random seed for reproducibility.
        history_days: Number of days of history to generate.
        start_date: Start date for the time series.
        spec: Hierarchy specification. Defaults to standard 224-node hierarchy.

    Returns:
        Tuple of (Y_df, S_df, X_future, X_past) — all Pandera-validated.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    if spec is None:
        spec = HierarchySpec()

    skus = generate_sku_ids(spec)
    dates = pd.date_range(start=start_date, periods=history_days, freq="D")
    n_dates = len(dates)

    # Determine intermittent SKUs (30% of total)
    n_intermittent = int(len(skus) * 0.30)
    intermittent_indices = set(rng.choice(len(skus), size=n_intermittent, replace=False))

    y_records = []
    x_future_records = []
    x_past_records = []

    holidays = _generate_holidays(dates)

    for sku_idx, sku in enumerate(skus):
        uid = sku["unique_id"]
        category = sku["category"]
        warehouse = sku["warehouse"]
        params = CATEGORY_PARAMS[category]

        # --- Base demand (Negative Binomial distribution) ---
        # NB parameterization: mean = base_demand_mean, variance = base_demand_var
        mu = params["base_demand_mean"]
        var = params["base_demand_var"]
        # Negative binomial: n = mu^2 / (var - mu), p = mu / var
        # NOTE: All current CATEGORY_PARAMS have var < mu (e.g. Electronics 8<15,
        # Grocery 30<80, etc.), so this NB branch is never reached. The logic is
        # correct and will activate if a future category has var > mu.
        if var > mu:
            n_param = mu ** 2 / (var - mu)
            p_param = mu / var
        else:
            n_param = mu
            p_param = 0.5
        base_demand = rng.negative_binomial(max(1, int(n_param)), min(0.99, max(0.01, p_param)), size=n_dates).astype(float)

        # --- Seasonality ---
        day_of_year = dates.dayofyear.values
        seasonality = 1 + params["seasonality_amp"] * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        # Day-of-week effect (weekends higher for retail)
        dow = dates.dayofweek.values
        dow_effect = np.where(dow >= 5, 1.15, 1.0)

        # --- Holiday boost ---
        holiday_boost = np.where(holidays == 1, 1.5, 1.0)

        # --- Trend (slight upward) ---
        trend = 1 + 0.0001 * np.arange(n_dates)

        # --- Price generation ---
        price_low, price_high = params["price_range"]
        base_price = rng.uniform(price_low, price_high)
        # Price varies with occasional promotions
        promo_flag = (rng.random(n_dates) < 0.08).astype(int)  # ~8% promo days
        price_discount = np.where(promo_flag == 1, rng.uniform(0.7, 0.9, n_dates), 1.0)
        price = base_price * price_discount
        # Random price fluctuation
        price = price * (1 + rng.normal(0, 0.02, n_dates))
        price = np.maximum(price, 1.0)

        # --- Price elasticity ---
        price_ratio = price / base_price
        elasticity_effect = 1 + params["elasticity"] * (price_ratio - 1) * 10

        # --- Combine all effects ---
        demand = base_demand * seasonality * dow_effect * holiday_boost * trend * elasticity_effect

        # --- Intermittent demand (M5 property 1) ---
        if sku_idx in intermittent_indices:
            zero_mask = rng.random(n_dates) < 0.55  # 55% zero days
            demand[zero_mask] = 0.0

        # --- Censored demand (M5 property 5) ---
        # Simulate stock levels and censor when stock = 0
        stock_level = np.zeros(n_dates)
        initial_stock = float(rng.integers(50, 500))
        reorder_point = mu * 7  # 1-week safety stock
        order_qty = mu * 14  # 2-week order
        lead_time = int(rng.integers(3, 10))
        pending_order_day = -1

        for t in range(n_dates):
            if t == 0:
                stock_level[t] = initial_stock
            else:
                stock_level[t] = stock_level[t - 1]

            # Receive pending order
            if pending_order_day >= 0 and t - pending_order_day >= lead_time:
                stock_level[t] += order_qty
                pending_order_day = -1

            # Censor demand if stock is zero
            if stock_level[t] <= 0:
                demand[t] = 0.0  # observed = 0, true demand was positive
                stock_level[t] = 0.0
            else:
                fulfilled = min(demand[t], stock_level[t])
                demand[t] = fulfilled
                stock_level[t] -= fulfilled

            # Reorder logic
            if stock_level[t] < reorder_point and pending_order_day < 0:
                pending_order_day = t

        # Ensure non-negative
        demand = np.maximum(demand, 0.0)

        # --- Temperature ---
        temperature = _generate_temperature(dates, warehouse, rng)

        # --- Build records ---
        for t in range(n_dates):
            y_records.append({"unique_id": uid, "ds": dates[t], "y": round(float(demand[t]), 2)})
            x_future_records.append(
                {
                    "unique_id": uid,
                    "ds": dates[t],
                    "promo_flag": int(promo_flag[t]),
                    "is_holiday": int(holidays[t]),
                    "temperature": round(float(temperature[t]), 1),
                }
            )
            x_past_records.append(
                {
                    "unique_id": uid,
                    "ds": dates[t],
                    "price": round(float(price[t]), 2),
                    "stock_level": round(float(stock_level[t]), 2),
                }
            )

    # --- Build DataFrames ---
    Y_df = pd.DataFrame(y_records)
    S_df = pd.DataFrame(skus)
    X_future = pd.DataFrame(x_future_records)
    X_past = pd.DataFrame(x_past_records)

    # --- Substitution effect (M5 property 4) ---
    # Cross-elasticity: when one SKU in a subcategory has a promo,
    # same-subcategory SKUs see reduced demand
    _apply_substitution_effects(Y_df, S_df, X_future, rng)

    # --- Validate with Pandera contracts ---
    Y_df = Y_SCHEMA.validate(Y_df)
    S_df = S_SCHEMA.validate(S_df)
    X_future = X_FUTURE_SCHEMA.validate(X_future)
    X_past = X_PAST_SCHEMA.validate(X_past)

    return Y_df, S_df, X_future, X_past


def _apply_substitution_effects(
    Y_df: pd.DataFrame,
    S_df: pd.DataFrame,
    X_future: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    """Apply cross-elasticity substitution effects within subcategories.

    When a SKU has a promotion, other SKUs in the same subcategory
    see a 3-8% demand reduction (substitution).

    Vectorized via pivot table operations to avoid O(n*m) inner loops.
    """
    # Group SKUs by subcategory
    subcat_groups: dict[str, list[str]] = {}
    for _, row in S_df.iterrows():
        key = f"{row['warehouse']}_{row['category']}_{row['subcategory']}"
        subcat_groups.setdefault(key, []).append(row["unique_id"])

    # Pivot demand and promo data for vectorized operations
    promo_pivot = X_future.pivot(index="ds", columns="unique_id", values="promo_flag")
    demand_pivot = Y_df.pivot(index="ds", columns="unique_id", values="y")

    for group_key, sku_ids in subcat_groups.items():
        if len(sku_ids) < 2:
            continue

        # Filter to group members present in both pivots
        members = [uid for uid in sku_ids if uid in promo_pivot.columns and uid in demand_pivot.columns]
        if len(members) < 2:
            continue

        promo_sub = promo_pivot[members]  # (n_dates, n_members)
        demand_sub = demand_pivot[members].copy()

        # For each SKU, when it has a promo, reduce *other* members' demand
        for uid in members:
            promo_mask = promo_sub[uid] == 1  # boolean series over dates
            if promo_mask.sum() == 0:
                continue
            other_members = [m for m in members if m != uid]
            n_promo_dates = int(promo_mask.sum())
            # Generate reduction factors (0.92-0.97) for all affected cells at once
            reduction = rng.uniform(0.92, 0.97, size=(n_promo_dates, len(other_members)))
            demand_sub.loc[promo_mask, other_members] = (
                demand_sub.loc[promo_mask, other_members].values * reduction
            )

        # Clip and round
        demand_sub = demand_sub.clip(lower=0).round(2)
        demand_pivot[members] = demand_sub

    # Unpivot back into Y_df long format
    melted = demand_pivot.reset_index().melt(id_vars="ds", var_name="unique_id", value_name="y")
    # Update Y_df in place by re-indexing
    Y_df.set_index(["unique_id", "ds"], inplace=True)
    melted.set_index(["unique_id", "ds"], inplace=True)
    Y_df.update(melted)
    Y_df.reset_index(inplace=True)


def build_hierarchy_matrix(S_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Build summation matrix S for hierarchical reconciliation.

    Returns:
        S: Summation matrix of shape (n_total_nodes, n_bottom_level)
        tags: Dictionary mapping level names to column values for aggregation.
    """
    sku_ids = S_df["unique_id"].tolist()
    n_bottom = len(sku_ids)
    sku_id_to_idx = {uid: idx for idx, uid in enumerate(sku_ids)}

    # Bottom level: identity
    S_bottom = np.eye(n_bottom)

    # Category level: warehouse × subcategory = 3 × 20 = 60
    cat_keys = []
    S_cat_rows = []
    for _, grp in S_df.groupby(["warehouse", "category", "subcategory"]):
        indices = [sku_id_to_idx[uid] for uid in grp["unique_id"]]
        row = np.zeros(n_bottom)
        row[indices] = 1.0
        S_cat_rows.append(row)
        cat_keys.append(f"{grp.iloc[0]['warehouse']}/{grp.iloc[0]['category']}/{grp.iloc[0]['subcategory']}")
    S_cat = np.array(S_cat_rows)

    # Warehouse level: 3
    warehouse_keys = []
    S_wh_rows = []
    for wh, grp in S_df.groupby("warehouse"):
        indices = [sku_id_to_idx[uid] for uid in grp["unique_id"]]
        row = np.zeros(n_bottom)
        row[indices] = 1.0
        S_wh_rows.append(row)
        warehouse_keys.append(str(wh))
    S_wh = np.array(S_wh_rows)

    # National level: 1
    S_national = np.ones((1, n_bottom))

    # Stack: national (1) + warehouse (3) + category (60) + sku (200)
    S = np.vstack([S_national, S_wh, S_cat, S_bottom])

    tags = {
        "national": ["Total"],
        "warehouse": warehouse_keys,
        "category": cat_keys,
        "sku": sku_ids,
    }

    return S, tags


def compute_data_hash(Y_df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of generated data for reproducibility verification."""
    content = Y_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def get_data_statistics(Y_df: pd.DataFrame, S_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics of the generated data."""
    n_skus = Y_df["unique_id"].nunique()
    n_days = Y_df["ds"].nunique()
    total_rows = len(Y_df)

    # Intermittent demand analysis
    zero_pct_per_sku = Y_df.groupby("unique_id")["y"].apply(lambda x: (x == 0).mean())
    n_intermittent = (zero_pct_per_sku > 0.5).sum()

    return {
        "total_rows": total_rows,
        "n_skus": n_skus,
        "n_days": n_days,
        "n_intermittent_skus": int(n_intermittent),
        "intermittent_pct": round(n_intermittent / max(n_skus, 1) * 100, 1),
        "mean_demand": round(float(Y_df["y"].mean()), 2),
        "median_demand": round(float(Y_df["y"].median()), 2),
        "zero_demand_pct": round(float((Y_df["y"] == 0).mean() * 100), 1),
        "n_warehouses": S_df["warehouse"].nunique(),
        "n_categories": S_df["category"].nunique(),
        "n_subcategories": S_df["subcategory"].nunique(),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    validate_only = "--validate-only" in sys.argv

    print("Generating demand data...")
    Y_df, S_df, X_future, X_past = generate_demand_data(seed=42)

    stats = get_data_statistics(Y_df, S_df)
    print(f"Generated {stats['total_rows']:,} rows for {stats['n_skus']} SKUs over {stats['n_days']} days")
    print(f"Intermittent SKUs: {stats['n_intermittent_skus']} ({stats['intermittent_pct']}%)")
    print(f"Zero demand: {stats['zero_demand_pct']}%")
    print(f"Mean demand: {stats['mean_demand']}")

    data_hash = compute_data_hash(Y_df)
    print(f"SHA-256: {data_hash}")

    S, tags = build_hierarchy_matrix(S_df)
    print(f"Hierarchy: S matrix shape {S.shape} ({sum(len(v) for v in tags.values())} nodes)")

    if validate_only:
        print("Validation passed.")
        sys.exit(0)
