"""MLflow model registry integration for ChainInsight.

Provides a ModelRegistry class for logging, registering, transitioning,
and retrieving models via MLflow's tracking and model registry APIs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Guard import — MLflow is optional
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.info("mlflow not installed — ModelRegistry will use no-op mode")


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@dataclass
class ModelRunInfo:
    """Metadata returned after logging a model run."""

    run_id: str
    experiment_id: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    artifact_uri: str = ""
    timestamp: str = ""


class ModelRegistry:
    """Unified interface for MLflow model lifecycle management.

    If MLflow is not installed, all methods degrade gracefully with warnings.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
        self._client: Any = None

        if HAS_MLFLOW:
            mlflow.set_tracking_uri(self._tracking_uri)
            self._client = MlflowClient(self._tracking_uri)
            logger.info("MLflow registry initialized at %s", self._tracking_uri)
        else:
            logger.warning("MLflow not available — registry operations will be no-ops")

    def _require_mlflow(self) -> bool:
        """Check MLflow availability, log warning if missing."""
        if not HAS_MLFLOW or self._client is None:
            logger.warning("MLflow is not available; skipping registry operation")
            return False
        return True

    def log_model_run(
        self,
        model_name: str,
        model_artifact: Any,
        metrics: dict[str, float],
        params: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        experiment_name: str = "chaininsight",
        artifact_path: str = "model",
    ) -> ModelRunInfo | None:
        """Log a model training run with metrics, params, and artifact.

        Args:
            model_name: Logical model name (e.g., 'lightgbm-forecaster').
            model_artifact: The trained model object (sklearn-compatible).
            metrics: Dictionary of metric name -> value (e.g., {'mape': 12.1}).
            params: Dictionary of hyperparameters.
            tags: Additional tags for the run.
            experiment_name: MLflow experiment name.
            artifact_path: Sub-path for model artifact storage.

        Returns:
            ModelRunInfo with run details, or None if MLflow is unavailable.
        """
        if not self._require_mlflow():
            return None

        params = params or {}
        tags = tags or {}

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            # Log parameters
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("logged_at", datetime.now(timezone.utc).isoformat())
            for k, v in tags.items():
                mlflow.set_tag(k, v)

            # Log model artifact
            try:
                mlflow.sklearn.log_model(model_artifact, artifact_path)
            except Exception:
                # Fall back to generic pickle logging
                try:
                    mlflow.pyfunc.log_model(artifact_path, python_model=model_artifact)
                except Exception:
                    logger.warning("Could not log model artifact for %s", model_name)

            info = ModelRunInfo(
                run_id=run.info.run_id,
                experiment_id=run.info.experiment_id,
                model_name=model_name,
                metrics=metrics,
                params=params,
                artifact_uri=run.info.artifact_uri,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            logger.info(
                "Logged model run: %s (run_id=%s, mape=%.2f)",
                model_name,
                info.run_id,
                metrics.get("mape", -1),
            )
            return info

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: str = "",
    ) -> str | None:
        """Register a logged model in the MLflow Model Registry.

        Args:
            run_id: The MLflow run ID containing the model artifact.
            model_name: Registry name for the model.
            artifact_path: Path within the run's artifacts.
            description: Optional description for the registered model.

        Returns:
            The model version string, or None if unavailable.
        """
        if not self._require_mlflow():
            return None

        model_uri = f"runs:/{run_id}/{artifact_path}"

        try:
            # Create registered model if it doesn't exist
            try:
                self._client.create_registered_model(model_name, description=description)
                logger.info("Created registered model: %s", model_name)
            except mlflow.exceptions.MlflowException:
                # Model already exists
                pass

            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            logger.info("Registered model %s version %s", model_name, version)
            return str(version)

        except Exception:
            logger.exception("Failed to register model %s from run %s", model_name, run_id)
            return None

    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> bool:
        """Transition a model version to a new stage.

        Args:
            model_name: Registered model name.
            version: Model version to transition.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing: Whether to archive existing models in the target stage.

        Returns:
            True if successful, False otherwise.
        """
        if not self._require_mlflow():
            return False

        valid_stages = {"Staging", "Production", "Archived", "None"}
        if stage not in valid_stages:
            logger.error("Invalid stage '%s'. Must be one of %s", stage, valid_stages)
            return False

        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(
                "Transitioned %s v%s to stage '%s' (archive_existing=%s)",
                model_name,
                version,
                stage,
                archive_existing,
            )
            return True

        except Exception:
            logger.exception(
                "Failed to transition %s v%s to %s", model_name, version, stage
            )
            return False

    def get_production_model(
        self,
        model_name: str,
    ) -> Any | None:
        """Load the current Production-stage model.

        Args:
            model_name: Registered model name.

        Returns:
            The loaded model object, or None if not found/available.
        """
        if not self._require_mlflow():
            return None

        try:
            # Find latest version in Production stage
            versions = self._client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                logger.warning("No Production version found for %s", model_name)
                return None

            latest = versions[0]
            model_uri = f"models:/{model_name}/{latest.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(
                "Loaded production model %s v%s (run_id=%s)",
                model_name,
                latest.version,
                latest.run_id,
            )
            return model

        except Exception:
            logger.exception("Failed to load production model %s", model_name)
            return None

    def list_models(
        self,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """List all registered models with their latest versions.

        Args:
            max_results: Maximum number of models to return.

        Returns:
            List of dictionaries with model metadata.
        """
        if not self._require_mlflow():
            return []

        try:
            registered_models = self._client.search_registered_models(
                max_results=max_results
            )

            results = []
            for rm in registered_models:
                latest_versions = []
                for mv in (rm.latest_versions or []):
                    latest_versions.append({
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "status": mv.status,
                        "run_id": mv.run_id,
                        "creation_timestamp": mv.creation_timestamp,
                    })

                results.append({
                    "name": rm.name,
                    "description": rm.description or "",
                    "creation_timestamp": rm.creation_timestamp,
                    "last_updated_timestamp": rm.last_updated_timestamp,
                    "latest_versions": latest_versions,
                })

            return results

        except Exception:
            logger.exception("Failed to list registered models")
            return []
