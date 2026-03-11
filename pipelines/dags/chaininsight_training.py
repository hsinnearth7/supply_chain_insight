"""ChainInsight ML Training DAG.

Orchestrates the full training pipeline:
validate_data -> generate_features -> [train_forecasters, train_rl] -> evaluate -> register -> promote

Schedule: Daily at 02:00 UTC
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "chaininsight",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    dag_id="chaininsight_training",
    default_args=default_args,
    description="ChainInsight ML model training pipeline",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["chaininsight", "ml", "training"],
)


def validate_data(**kwargs):
    """Validate raw data using Pandera contracts."""
    import pandas as pd
    from app.forecasting.contracts import validate_Y_df
    from app.settings import get_data_config

    config = get_data_config()
    data_path = config.get("raw_dir", "data/raw")

    Y_df = pd.read_parquet(f"{data_path}/Y_df.parquet")
    validate_Y_df(Y_df)

    record_count = len(Y_df)
    unique_ids = Y_df["unique_id"].nunique()
    kwargs["ti"].xcom_push(key="record_count", value=record_count)
    kwargs["ti"].xcom_push(key="unique_ids", value=unique_ids)
    print(f"Validated {record_count} records across {unique_ids} SKUs")


def generate_features(**kwargs):
    """Materialize features in the feature store."""
    from app.forecasting.feature_store import FeatureStore

    store = FeatureStore()
    store.materialize()
    print("Feature store materialized successfully")


def train_forecasters(**kwargs):
    """Train all forecasting models (LightGBM, XGBoost, SARIMAX, Chronos)."""
    import json

    from app.forecasting.models import ForecastModelFactory

    ti = kwargs["ti"]
    models_to_train = ["naive_ma30", "sarimax", "xgboost", "lightgbm"]

    results = {}
    for model_name in models_to_train:
        print(f"Training {model_name}...")
        model = ForecastModelFactory.create(model_name)
        metrics = model.train()
        results[model_name] = metrics
        print(f"  {model_name} -> MAPE: {metrics.get('mape', 'N/A')}")

    ti.xcom_push(key="forecast_results", value=json.dumps(results, default=str))


def train_rl(**kwargs):
    """Train RL inventory optimization agent (PPO)."""
    import json

    ti = kwargs["ti"]

    from app.rl.curriculum import CurriculumTrainer

    trainer = CurriculumTrainer()
    metrics = trainer.train()

    ti.xcom_push(key="rl_results", value=json.dumps(metrics, default=str))
    print(f"RL training complete — final cost: {metrics.get('final_cost', 'N/A')}")


def evaluate(**kwargs):
    """Evaluate trained models using walk-forward cross-validation."""
    import json

    ti = kwargs["ti"]

    forecast_results = json.loads(ti.xcom_pull(key="forecast_results") or "{}")
    rl_results = json.loads(ti.xcom_pull(key="rl_results") or "{}")

    from app.forecasting.evaluation import WalkForwardEvaluator

    evaluator = WalkForwardEvaluator()
    eval_report = evaluator.evaluate()

    combined = {
        "forecast_training": forecast_results,
        "rl_training": rl_results,
        "evaluation": eval_report,
    }

    ti.xcom_push(key="eval_report", value=json.dumps(combined, default=str))

    best_mape = eval_report.get("best_mape", float("inf"))
    print(f"Evaluation complete — best MAPE: {best_mape:.2f}%")

    # Gate: fail if MAPE exceeds threshold
    if best_mape > 25.0:
        raise ValueError(f"Best MAPE {best_mape:.2f}% exceeds threshold of 25%")


def register(**kwargs):
    """Register evaluated models in MLflow registry."""
    import json

    ti = kwargs["ti"]
    eval_report = json.loads(ti.xcom_pull(key="eval_report") or "{}")

    from app.mlflow_registry import ModelRegistry

    registry = ModelRegistry()

    forecast_results = eval_report.get("forecast_training", {})
    for model_name, metrics in forecast_results.items():
        if isinstance(metrics, dict):
            run_info = registry.log_model_run(
                model_name=f"chaininsight-{model_name}",
                model_artifact=None,
                metrics=metrics,
                params={"model_type": model_name},
                tags={"pipeline": "airflow", "dag": "chaininsight_training"},
            )
            if run_info:
                version = registry.register_model(
                    run_id=run_info.run_id,
                    model_name=f"chaininsight-{model_name}",
                )
                ti.xcom_push(key=f"version_{model_name}", value=version)
                print(f"Registered {model_name} v{version}")


def promote(**kwargs):
    """Promote best model to Production stage."""
    import json

    ti = kwargs["ti"]
    eval_report = json.loads(ti.xcom_pull(key="eval_report") or "{}")

    from app.mlflow_registry import ModelRegistry

    registry = ModelRegistry()

    # Find best forecasting model
    forecast_results = eval_report.get("forecast_training", {})
    best_model = None
    best_mape = float("inf")

    for model_name, metrics in forecast_results.items():
        if isinstance(metrics, dict):
            mape = metrics.get("mape", float("inf"))
            if mape < best_mape:
                best_mape = mape
                best_model = model_name

    if best_model:
        version = ti.xcom_pull(key=f"version_{best_model}")
        if version:
            registry.transition_stage(
                model_name=f"chaininsight-{best_model}",
                version=version,
                stage="Production",
                archive_existing=True,
            )
            print(f"Promoted {best_model} v{version} to Production (MAPE: {best_mape:.2f}%)")
        else:
            print(f"No version found for {best_model}, skipping promotion")
    else:
        print("No model to promote")


# Task definitions
t_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

t_features = PythonOperator(
    task_id="generate_features",
    python_callable=generate_features,
    dag=dag,
)

t_forecast = PythonOperator(
    task_id="train_forecasters",
    python_callable=train_forecasters,
    dag=dag,
)

t_rl = PythonOperator(
    task_id="train_rl",
    python_callable=train_rl,
    dag=dag,
)

t_evaluate = PythonOperator(
    task_id="evaluate",
    python_callable=evaluate,
    dag=dag,
)

t_register = PythonOperator(
    task_id="register",
    python_callable=register,
    dag=dag,
)

t_promote = PythonOperator(
    task_id="promote",
    python_callable=promote,
    dag=dag,
)

# DAG dependencies
t_validate >> t_features >> [t_forecast, t_rl] >> t_evaluate >> t_register >> t_promote
