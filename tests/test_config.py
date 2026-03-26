"""Tests for configuration management.

Covers: YAML loading, config accessors, defaults.
"""

import pytest

from app.settings import (
    get_capacity_config,
    get_data_config,
    get_eval_config,
    get_model_config,
    get_monitoring_config,
    get_sensing_config,
    get_sop_config,
    get_supply_chain_config,
    load_config,
)


class TestConfigLoading:
    def test_load_config_returns_dict(self):
        load_config.cache_clear()
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_data_section(self):
        load_config.cache_clear()
        config = load_config()
        assert "data" in config

    def test_config_has_model_section(self):
        load_config.cache_clear()
        config = load_config()
        assert "model" in config

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config.cache_clear()
            load_config("/nonexistent/path.yaml")


class TestDataConfig:
    def test_n_skus(self):
        config = get_data_config()
        assert isinstance(config.get("n_skus"), int) and config["n_skus"] > 0

    def test_n_warehouses(self):
        config = get_data_config()
        assert isinstance(config.get("n_warehouses"), int) and config["n_warehouses"] > 0

    def test_seed(self):
        config = get_data_config()
        assert isinstance(config.get("seed"), int)


class TestModelConfig:
    def test_default_model(self):
        config = get_model_config()
        assert config.get("default") in (
            "lightgbm", "xgboost", "sarimax", "chronos", "naive", "routing_ensemble",
        )

    def test_lightgbm_config(self):
        config = get_model_config("lightgbm")
        assert "n_estimators" in config
        assert "num_leaves" in config

    def test_routing_threshold(self):
        config = get_model_config("routing")
        assert config.get("cold_start_threshold_days") == 60


class TestEvalConfig:
    def test_cv_windows(self):
        config = get_eval_config()
        assert isinstance(config.get("cv_windows"), int) and config["cv_windows"] > 0

    def test_significance_alpha(self):
        config = get_eval_config()
        assert config.get("significance_alpha") == 0.05


class TestMonitoringConfig:
    def test_drift_thresholds(self):
        config = get_monitoring_config()
        assert config.get("drift_threshold_ks") == 0.05
        assert config.get("drift_threshold_psi") == 0.1

    def test_retrain_trigger(self):
        config = get_monitoring_config()
        assert config.get("retrain_trigger_days") == 7
        assert config.get("mape_alert_threshold") == 0.20


class TestCapacityConfig:
    def test_production_lines(self):
        config = get_capacity_config()
        assert config.get("production_lines") == 3

    def test_utilization_target(self):
        config = get_capacity_config()
        assert config.get("utilization_target") == 0.85

    def test_planning_horizon(self):
        config = get_capacity_config()
        assert config.get("planning_horizon_days") == 90


class TestSensingConfig:
    def test_signal_sources(self):
        config = get_sensing_config()
        assert isinstance(config.get("signal_sources"), list) and len(config.get("signal_sources")) > 0

    def test_spike_threshold(self):
        config = get_sensing_config()
        assert config.get("spike_threshold_sigma") == 2.5

    def test_weights_sum_to_one(self):
        config = get_sensing_config()
        total = config["pos_weight"] + config["social_weight"] + config["weather_weight"]
        assert abs(total - 1.0) < 1e-9


class TestSOPConfig:
    def test_default_scenarios(self):
        config = get_sop_config()
        assert isinstance(config.get("default_scenarios"), list) and "baseline" in config.get("default_scenarios")

    def test_target_fill_rate(self):
        config = get_sop_config()
        assert config.get("target_fill_rate") == 0.95

    def test_time_bucket(self):
        config = get_sop_config()
        assert config.get("time_bucket") == "weekly"


class TestSupplyChainConfig:
    def test_ordering_cost(self):
        config = get_supply_chain_config()
        assert config.get("ordering_cost") == 50
