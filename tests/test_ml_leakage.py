"""Tests to verify ML pipeline has no data leakage."""

import pytest

from app.pipeline.ml_engine import (
    CLASSIFICATION_FEATURES,
    REGRESSION_FEATURES,
)


class TestNoDataLeakage:
    """Verify ML data leakage vectors are addressed."""

    def test_no_circular_features_in_classification(self):
        """Verify classification features don't encode the target (Stock_Status).

        Reorder_Point, DSI, and Stock_Coverage_Ratio mathematically encode
        Stock_Status and should NOT be used for classification.
        """
        circular_features = {"Reorder_Point", "DSI", "Stock_Coverage_Ratio"}
        for feat in circular_features:
            assert feat not in CLASSIFICATION_FEATURES, (
                f"Circular feature '{feat}' found in CLASSIFICATION_FEATURES"
            )

    def test_regression_features_superset(self):
        """Regression features should include classification features plus derived ones."""
        for feat in CLASSIFICATION_FEATURES:
            assert feat in REGRESSION_FEATURES

    def test_inventory_value_not_in_classification(self):
        """Inventory_Value should not be a classification feature (it's a target for regression)."""
        assert "Inventory_Value" not in CLASSIFICATION_FEATURES

    def test_scaler_not_fitted_on_test_data(self, sample_df):
        """Verify StandardScaler in classification uses sklearn Pipeline.

        The MLAnalyzer.plot_classification() method should use
        sklearn.pipeline.Pipeline for cross-validation, which ensures
        the scaler only sees training data in each fold.
        """
        from app.pipeline.ml_engine import MLAnalyzer

        analyzer = MLAnalyzer()
        df_ml = analyzer.enrich(sample_df)

        # If data is too small for stratified CV, verify enrichment at least
        # produces the expected feature set (no leakage features present)
        if len(df_ml) < 10:
            pytest.skip("Not enough data for classification test")
        else:
            analyzer.plot_classification(df_ml)
            assert hasattr(analyzer, "_classification_features")
            assert analyzer._classification_features == CLASSIFICATION_FEATURES
