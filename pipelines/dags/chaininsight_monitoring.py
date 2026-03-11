"""ChainInsight Drift Monitoring DAG.

Runs every 6 hours to detect data drift, prediction drift, and concept drift.
Triggers alerts and auto-retrain if thresholds are exceeded.

Schedule: Every 6 hours
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

default_args = {
    "owner": "chaininsight",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "execution_timeout": timedelta(minutes=30),
}

dag = DAG(
    dag_id="chaininsight_monitoring",
    default_args=default_args,
    description="ChainInsight drift monitoring and auto-retrain triggers",
    schedule_interval="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["chaininsight", "monitoring", "drift"],
)


def check_data_drift(**kwargs):
    """Run KS-test data drift detection on feature distributions."""
    import json

    ti = kwargs["ti"]
    drift_results = {
        "data_drift": {"detected": False, "features_drifted": [], "ks_stats": {}},
    }

    try:
        from app.forecasting.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        report = monitor.check_data_drift()

        drift_results["data_drift"] = {
            "detected": report.get("drift_detected", False),
            "features_drifted": report.get("drifted_features", []),
            "ks_stats": report.get("ks_statistics", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }

        if drift_results["data_drift"]["detected"]:
            print(f"DATA DRIFT detected in features: {report.get('drifted_features', [])}")
            # Track in Prometheus
            try:
                from app.metrics import track_drift
                track_drift("data")
            except ImportError:
                pass
        else:
            print("No data drift detected")

    except Exception as e:
        print(f"Data drift check failed: {e}")
        drift_results["data_drift"]["error"] = str(e)

    ti.xcom_push(key="data_drift", value=json.dumps(drift_results, default=str))


def check_prediction_drift(**kwargs):
    """Run PSI prediction drift detection."""
    import json

    ti = kwargs["ti"]
    drift_results = {
        "prediction_drift": {"detected": False, "psi_value": 0.0},
    }

    try:
        from app.forecasting.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        report = monitor.check_prediction_drift()

        drift_results["prediction_drift"] = {
            "detected": report.get("drift_detected", False),
            "psi_value": report.get("psi", 0.0),
            "threshold": 0.1,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if drift_results["prediction_drift"]["detected"]:
            print(f"PREDICTION DRIFT detected — PSI: {report.get('psi', 0):.4f}")
            try:
                from app.metrics import track_drift
                track_drift("prediction")
            except ImportError:
                pass
        else:
            print("No prediction drift detected")

    except Exception as e:
        print(f"Prediction drift check failed: {e}")
        drift_results["prediction_drift"]["error"] = str(e)

    ti.xcom_push(key="prediction_drift", value=json.dumps(drift_results, default=str))


def check_concept_drift(**kwargs):
    """Check MAPE trend for concept drift (>20% for 7 consecutive days)."""
    import json

    ti = kwargs["ti"]
    drift_results = {
        "concept_drift": {"detected": False, "mape_trend": []},
    }

    try:
        from app.forecasting.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        report = monitor.check_concept_drift()

        drift_results["concept_drift"] = {
            "detected": report.get("drift_detected", False),
            "mape_trend": report.get("mape_history", []),
            "consecutive_days_above_threshold": report.get("consecutive_days", 0),
            "threshold_pct": 20.0,
            "threshold_days": 7,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if drift_results["concept_drift"]["detected"]:
            print(f"CONCEPT DRIFT detected — {report.get('consecutive_days', 0)} days above 20% MAPE")
            try:
                from app.metrics import track_drift
                track_drift("concept")
            except ImportError:
                pass
        else:
            print("No concept drift detected")

    except Exception as e:
        print(f"Concept drift check failed: {e}")
        drift_results["concept_drift"]["error"] = str(e)

    ti.xcom_push(key="concept_drift", value=json.dumps(drift_results, default=str))


def decide_retrain(**kwargs):
    """Decide whether to trigger retraining based on drift results."""
    import json

    ti = kwargs["ti"]

    data_drift = json.loads(ti.xcom_pull(key="data_drift") or "{}")
    pred_drift = json.loads(ti.xcom_pull(key="prediction_drift") or "{}")
    concept_drift = json.loads(ti.xcom_pull(key="concept_drift") or "{}")

    any_drift = (
        data_drift.get("data_drift", {}).get("detected", False)
        or pred_drift.get("prediction_drift", {}).get("detected", False)
        or concept_drift.get("concept_drift", {}).get("detected", False)
    )

    if any_drift:
        print("Drift detected — triggering retrain")
        return "trigger_retrain"
    else:
        print("No drift detected — skipping retrain")
        return "skip_retrain"


def trigger_retrain(**kwargs):
    """Trigger the training DAG for retraining."""
    from airflow.api.common.trigger_dag import trigger_dag as airflow_trigger_dag

    airflow_trigger_dag(
        dag_id="chaininsight_training",
        conf={"trigger_reason": "drift_detected", "triggered_by": "monitoring_dag"},
    )
    print("Triggered chaininsight_training DAG for retraining")


def skip_retrain(**kwargs):
    """No-op task when no drift is detected."""
    print("No retraining needed — all drift checks passed")


# Task definitions
t_data_drift = PythonOperator(
    task_id="check_data_drift",
    python_callable=check_data_drift,
    dag=dag,
)

t_pred_drift = PythonOperator(
    task_id="check_prediction_drift",
    python_callable=check_prediction_drift,
    dag=dag,
)

t_concept_drift = PythonOperator(
    task_id="check_concept_drift",
    python_callable=check_concept_drift,
    dag=dag,
)

t_decide = BranchPythonOperator(
    task_id="decide_retrain",
    python_callable=decide_retrain,
    dag=dag,
)

t_retrain = PythonOperator(
    task_id="trigger_retrain",
    python_callable=trigger_retrain,
    dag=dag,
)

t_skip = PythonOperator(
    task_id="skip_retrain",
    python_callable=skip_retrain,
    dag=dag,
)

# DAG dependencies — drift checks run in parallel, then decide
[t_data_drift, t_pred_drift, t_concept_drift] >> t_decide >> [t_retrain, t_skip]
