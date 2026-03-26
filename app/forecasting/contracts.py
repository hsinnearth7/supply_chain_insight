"""Data contracts for ChainInsight forecasting pipeline.

All DataFrames entering or exiting the pipeline must pass these Pandera schemas.
Contract violation → pipeline halt + alert.

Schemas:
    Y_SCHEMA:       Demand time series (unique_id, ds, y)
    S_SCHEMA:       Static attributes (unique_id, warehouse, category, subcategory)
    X_FUTURE_SCHEMA: Known future exogenous (promo_flag, is_holiday, temperature)
    X_PAST_SCHEMA:   Historical dynamic exogenous (price, stock_level)
    FORECAST_SCHEMA: Model output (unique_id, ds, y_hat, y_lo, y_hi, model)
"""

from __future__ import annotations

import pandera as pa

# Re-export from data_generator for backward compatibility
# TODO: Inverted dependency — contracts.py (the canonical schema owner) imports
# schemas FROM data_generator.py. The dependency direction should be reversed:
# schemas should be defined here and data_generator should import from contracts.
# Left as-is to avoid breaking existing imports.
from app.forecasting.data_generator import (
    S_SCHEMA,
    X_FUTURE_SCHEMA,
    X_PAST_SCHEMA,
    Y_SCHEMA,
)

# Additional schema for forecast outputs
FORECAST_SCHEMA = pa.DataFrameSchema(
    {
        "unique_id": pa.Column(str, nullable=False),
        "ds": pa.Column("datetime64[ns]", nullable=False),
        "y_hat": pa.Column(float, nullable=False),
        "y_lo_90": pa.Column(float, nullable=True),
        "y_hi_90": pa.Column(float, nullable=True),
        "model": pa.Column(str, nullable=False),
    },
    strict=False,  # allow extra columns
    coerce=True,
)

# Schema for evaluation results
EVAL_SCHEMA = pa.DataFrameSchema(
    {
        "model": pa.Column(str, nullable=False),
        "mape": pa.Column(float, pa.Check.ge(0), nullable=True),
        "rmse": pa.Column(float, pa.Check.ge(0), nullable=False),
        "mae": pa.Column(float, pa.Check.ge(0), nullable=False),
    },
    strict=False,
    coerce=True,
)

__all__ = [
    "Y_SCHEMA",
    "S_SCHEMA",
    "X_FUTURE_SCHEMA",
    "X_PAST_SCHEMA",
    "FORECAST_SCHEMA",
    "EVAL_SCHEMA",
]
