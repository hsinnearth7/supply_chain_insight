"""Prometheus metrics instrumentation for ChainInsight.

Provides counters, histograms, and gauges for HTTP requests, forecasting
accuracy, RL optimization costs, drift detection, and pipeline performance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Guard import — prometheus_client is optional
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.info("prometheus_client not installed — metrics will be no-ops")

# ---------------------------------------------------------------------------
# Registry and metric definitions
# ---------------------------------------------------------------------------

if HAS_PROMETHEUS:
    REGISTRY = CollectorRegistry()

    # HTTP metrics
    HTTP_REQUESTS_TOTAL = Counter(
        "chaininsight_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
        registry=REGISTRY,
    )

    HTTP_REQUEST_DURATION = Histogram(
        "chaininsight_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=REGISTRY,
    )

    # Forecasting metrics
    FORECAST_MAPE = Gauge(
        "chaininsight_forecast_mape",
        "Forecast MAPE per product",
        ["product_id", "model"],
        registry=REGISTRY,
    )

    # RL metrics
    RL_COST_PER_DAY = Gauge(
        "chaininsight_rl_cost_per_day",
        "RL optimization cost per day (USD)",
        registry=REGISTRY,
    )

    # Drift metrics
    DRIFT_DETECTED = Counter(
        "chaininsight_drift_detected_total",
        "Drift detection events",
        ["drift_type"],
        registry=REGISTRY,
    )

    # Pipeline metrics
    PIPELINE_DURATION = Histogram(
        "chaininsight_pipeline_duration_seconds",
        "Pipeline stage duration in seconds",
        ["stage"],
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
        registry=REGISTRY,
    )

    # Feature store metrics
    FEATURE_STORE_LATENCY = Histogram(
        "chaininsight_feature_store_latency_seconds",
        "Feature store query latency in seconds",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        registry=REGISTRY,
    )

    # Error metrics
    ERRORS_TOTAL = Counter(
        "chaininsight_errors_total",
        "Total error count",
        ["error_type"],
        registry=REGISTRY,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_metrics_response() -> tuple[str, str]:
    """Generate Prometheus metrics response.

    Returns:
        Tuple of (metrics_text, content_type).
    """
    if not HAS_PROMETHEUS:
        return "# prometheus_client not installed\n", "text/plain"

    metrics_output = generate_latest(REGISTRY)
    return metrics_output.decode("utf-8"), "text/plain; version=0.0.4; charset=utf-8"


def track_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Record an HTTP request.

    Args:
        method: HTTP method (GET, POST, etc.).
        endpoint: Request path.
        status_code: HTTP response status code.
        duration: Request duration in seconds.
    """
    if not HAS_PROMETHEUS:
        return

    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code),
    ).inc()

    HTTP_REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint,
    ).observe(duration)


def track_forecast(product_id: str, model: str, mape: float) -> None:
    """Record forecast accuracy for a product.

    Args:
        product_id: SKU identifier.
        model: Model name used for forecasting.
        mape: Mean Absolute Percentage Error.
    """
    if not HAS_PROMETHEUS:
        return

    FORECAST_MAPE.labels(product_id=product_id, model=model).set(mape)


def track_rl_cost(cost: float) -> None:
    """Record RL daily optimization cost.

    Args:
        cost: Total daily inventory cost in USD.
    """
    if not HAS_PROMETHEUS:
        return

    RL_COST_PER_DAY.set(cost)


def track_drift(drift_type: str) -> None:
    """Increment drift detection counter.

    Args:
        drift_type: Type of drift detected ('data', 'prediction', 'concept').
    """
    if not HAS_PROMETHEUS:
        return

    DRIFT_DETECTED.labels(drift_type=drift_type).inc()


def track_pipeline(stage: str, duration: float) -> None:
    """Record pipeline stage duration.

    Args:
        stage: Pipeline stage name.
        duration: Duration in seconds.
    """
    if not HAS_PROMETHEUS:
        return

    PIPELINE_DURATION.labels(stage=stage).observe(duration)


def track_feature_store(operation: str, duration: float) -> None:
    """Record feature store operation latency.

    Args:
        operation: Operation type ('read', 'write', 'materialize').
        duration: Duration in seconds.
    """
    if not HAS_PROMETHEUS:
        return

    FEATURE_STORE_LATENCY.labels(operation=operation).observe(duration)


def track_error(error_type: str) -> None:
    """Increment error counter.

    Args:
        error_type: Category of error ('forecast', 'rl', 'pipeline', 'api').
    """
    if not HAS_PROMETHEUS:
        return

    ERRORS_TOTAL.labels(error_type=error_type).inc()


class MetricsTimer:
    """Context manager for timing operations and recording to Prometheus.

    Usage:
        with MetricsTimer('pipeline', 'etl'):
            run_etl()
    """

    def __init__(self, metric_type: str, label: str) -> None:
        self._metric_type = metric_type
        self._label = label
        self._start: float = 0

    def __enter__(self) -> MetricsTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        duration = time.perf_counter() - self._start
        if self._metric_type == "pipeline":
            track_pipeline(self._label, duration)
        elif self._metric_type == "feature_store":
            track_feature_store(self._label, duration)
