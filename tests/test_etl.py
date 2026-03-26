"""ETL pipeline tests."""

import os
import tempfile

import pytest

from app.pipeline.etl import ETLPipeline


class TestETLPipeline:
    """Test the 8-step ETL pipeline."""

    def test_etl_produces_correct_columns(self, sample_csv_file):
        """Verify ETL produces expected 11 output columns."""
        etl = ETLPipeline()
        df = etl.run(sample_csv_file)
        expected_cols = [
            "Product_ID", "Category", "Unit_Cost", "Current_Stock",
            "Daily_Demand_Est", "Safety_Stock_Target", "Vendor_Name",
            "Lead_Time_Days", "Reorder_Point", "Stock_Status", "Inventory_Value",
        ]
        assert list(df.columns) == expected_cols

    def test_etl_cleans_dirty_data(self, sample_csv_file):
        """Verify ETL produces expected output from known dirty input."""
        etl = ETLPipeline()
        df = etl.run(sample_csv_file)

        # Cost should be extracted from "$50.00" format
        p001 = df[df["Product_ID"] == "P001"].iloc[0]
        assert p001["Unit_Cost"] == 50.0

        # Negative stock should be clamped to 0
        p004 = df[df["Product_ID"] == "P004"].iloc[0]
        assert p004["Current_Stock"] >= 0

        # Out of Stock for zero stock
        p002 = df[df["Product_ID"] == "P002"].iloc[0]
        assert p002["Stock_Status"] == "Out of Stock"

        # Inventory_Value = stock * cost
        assert p001["Inventory_Value"] == p001["Current_Stock"] * p001["Unit_Cost"]

    def test_etl_handles_empty_csv(self):
        """Verify graceful handling of empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("")
            path = f.name

        try:
            etl = ETLPipeline()
            with pytest.raises(ValueError, match="empty"):
                etl.run(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_etl_validates_schema(self):
        """Verify missing columns are caught."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("A,B,C\n1,2,3\n")
            path = f.name

        try:
            etl = ETLPipeline()
            with pytest.raises(ValueError, match="missing required columns"):
                etl.run(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_etl_stats(self, sample_csv_file):
        """Verify pipeline statistics are captured."""
        etl = ETLPipeline()
        etl.run(sample_csv_file)
        stats = etl.get_stats()
        assert "raw_rows" in stats
        assert stats["raw_rows"] == 5
        assert "out_of_stock" in stats

    def test_etl_derived_fields(self, sample_df):
        """Verify Reorder_Point, Stock_Status, Inventory_Value are computed."""
        assert "Reorder_Point" in sample_df.columns
        assert "Stock_Status" in sample_df.columns
        assert "Inventory_Value" in sample_df.columns
        assert set(sample_df["Stock_Status"].unique()).issubset(
            {"Normal Stock", "Low Stock", "Out of Stock"}
        )
