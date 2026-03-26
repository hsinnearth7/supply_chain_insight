"""ETL Pipeline — 8-step data cleaning for supply chain inventory data."""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ETLPipeline:
    """8-step ETL pipeline that transforms dirty inventory CSV into clean data."""

    def __init__(self):
        self.stats = {}

    def run(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """Run the full ETL pipeline.

        Args:
            input_path: Path to dirty CSV file.
            output_path: Optional path to save cleaned CSV.

        Returns:
            Cleaned DataFrame with 11 columns.
        """
        logger.info("ETL pipeline started")
        self.stats = {}
        df = self._extract(input_path)
        df = self._transform(df)
        if output_path:
            self._load(df, output_path)
        logger.info("ETL pipeline completed — %d rows, %d columns", len(df), len(df.columns))
        return df

    def _load(self, df: pd.DataFrame, output_path: str):
        """Save cleaned DataFrame to CSV."""
        df.to_csv(output_path, index=False)
        logger.info("Saved cleaned data to %s (%d rows)", output_path, len(df))

    def run_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run transform steps on an already-loaded DataFrame."""
        df = df.copy()
        df = self._transform(df)
        return df

    # ------------------------------------------------------------------
    # Extract
    # ------------------------------------------------------------------
    REQUIRED_COLUMNS = {
        "Product_ID", "Category", "Unit_Cost_Raw", "Current_Stock_Raw",
        "Daily_Demand_Est", "Safety_Stock_Target", "Vendor_Name", "Lead_Time_Days",
    }

    def _extract(self, path: str) -> pd.DataFrame:
        logger.info("Step 0: Loading data from %s", path)
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}") from e
        if df.empty:
            raise ValueError(f"CSV file contains no data rows: {path}")
        # Schema validation
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        self.stats["raw_rows"] = len(df)
        self.stats["raw_columns"] = len(df.columns)
        return df

    # ------------------------------------------------------------------
    # Transform (8 steps)
    # ------------------------------------------------------------------
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._step1_clean_product_id(df)
        df = self._step2_clean_category(df)
        df = self._step3_clean_cost(df)
        df = self._step4_clean_stock(df)
        df = self._step5_handle_nulls(df)
        df = self._step6_clean_vendor(df)
        df = self._step7_validate(df)
        df = self._step8_derived_fields(df)
        return df

    def _step1_clean_product_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from Product_ID."""
        df["Product_ID"] = df["Product_ID"].str.strip()
        logger.info("Step 1: Product_ID cleaned")
        return df

    def _step2_clean_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize category casing."""
        before = df["Category"].nunique()
        df["Category"] = df["Category"].str.strip().str.capitalize()
        after = df["Category"].nunique()
        self.stats["categories_before"] = before
        self.stats["categories_after"] = after
        logger.info("Step 2: Category standardized (%d -> %d unique)", before, after)
        return df

    def _step3_clean_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric values from mixed cost formats."""
        def _parse(value):
            if pd.isna(value):
                return np.nan
            s = str(value).strip()
            if not any(c.isdigit() for c in s):
                return np.nan
            cleaned = re.sub(r"[^\d.]", "", s)
            try:
                return float(cleaned)
            except ValueError:
                return np.nan

        df["Unit_Cost"] = df["Unit_Cost_Raw"].apply(_parse)
        self.stats["invalid_costs"] = int(df["Unit_Cost"].isna().sum())
        logger.info("Step 3: Cost extracted (%d invalid)", self.stats["invalid_costs"])
        return df

    def _step4_clean_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce stock to numeric, clamp negatives to 0."""
        df["Current_Stock"] = pd.to_numeric(df["Current_Stock_Raw"], errors="coerce")
        neg_count = int((df["Current_Stock"] < 0).sum())
        df.loc[df["Current_Stock"] < 0, "Current_Stock"] = 0
        self.stats["negative_stocks"] = neg_count
        logger.info("Step 4: Stock cleaned (%d negatives -> 0)", neg_count)
        return df

    def _step5_handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill nulls: stock->0, cost->category median then global median."""
        null_stock = int(df["Current_Stock"].isna().sum())
        df["Current_Stock"] = df["Current_Stock"].fillna(0)

        null_cost = int(df["Unit_Cost"].isna().sum())
        df["Unit_Cost"] = df.groupby("Category")["Unit_Cost"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Unit_Cost"] = df["Unit_Cost"].fillna(df["Unit_Cost"].median())

        self.stats["null_stocks_filled"] = null_stock
        self.stats["null_costs_filled"] = null_cost
        logger.info("Step 5: Nulls filled (stock=%d, cost=%d)", null_stock, null_cost)
        return df

    def _step6_clean_vendor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace from Vendor_Name."""
        df["Vendor_Name"] = df["Vendor_Name"].str.strip()
        logger.info("Step 6: Vendor_Name cleaned")
        return df

    def _step7_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip numeric fields to valid ranges."""
        df["Daily_Demand_Est"] = df["Daily_Demand_Est"].clip(lower=0)
        df["Safety_Stock_Target"] = df["Safety_Stock_Target"].clip(lower=0)
        bad_lt = int((df["Lead_Time_Days"] < 1).sum())
        df["Lead_Time_Days"] = df["Lead_Time_Days"].clip(lower=1)
        self.stats["bad_lead_times"] = bad_lt
        logger.info("Step 7: Validation done (%d bad lead times)", bad_lt)
        return df

    def _step8_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Reorder_Point, Stock_Status, Inventory_Value."""
        df["Reorder_Point"] = (
            df["Daily_Demand_Est"] * df["Lead_Time_Days"] + df["Safety_Stock_Target"]
        )
        conditions = [
            df["Current_Stock"] == 0,
            df["Current_Stock"] < df["Reorder_Point"],
        ]
        choices = ["Out of Stock", "Low Stock"]
        df["Stock_Status"] = np.select(conditions, choices, default="Normal Stock")
        df["Inventory_Value"] = df["Current_Stock"] * df["Unit_Cost"]

        self.stats["out_of_stock"] = int((df["Stock_Status"] == "Out of Stock").sum())
        self.stats["low_stock"] = int((df["Stock_Status"] == "Low Stock").sum())
        self.stats["normal_stock"] = int((df["Stock_Status"] == "Normal Stock").sum())
        self.stats["total_inventory_value"] = float(df["Inventory_Value"].sum())
        logger.info("Step 8: Derived fields added")

        # Select output columns
        output_columns = [
            "Product_ID", "Category", "Unit_Cost", "Current_Stock",
            "Daily_Demand_Est", "Safety_Stock_Target", "Vendor_Name",
            "Lead_Time_Days", "Reorder_Point", "Stock_Status", "Inventory_Value",
        ]
        return df[output_columns]

    def get_stats(self) -> dict:
        """Return pipeline statistics for storage/display."""
        return self.stats.copy()
