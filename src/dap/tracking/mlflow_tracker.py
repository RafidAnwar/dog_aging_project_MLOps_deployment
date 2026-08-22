import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import PredictionErrorDisplay

from dap.config import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from dap.models.cslb_model import CSLBTrainingResult


def configure_mlflow() -> None:
    MLFLOW_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME,
    )

    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACT_ROOT.resolve().as_uri(),
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def save_feature_schema(
    feature_names: list[str],
    output_path: Path,
) -> None:
    output_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "target_column": "cslb_score",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def create_prediction_plot(
    y_true: pd.Series,
    y_pred,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))

    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        ax=axis,
    )

    axis.set_title("CSLB Prediction Error")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def log_training_run(
    training_result: CSLBTrainingResult,
    dataset_path: str | Path,
    test_size: float,
    random_state: int,
    output_dir: str | Path,
    y_test: pd.Series,
    y_pred,
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_schema_path = output_path / "feature_schema.json"
    plot_path = output_path / "prediction_error.png"

    save_feature_schema(
        feature_names=training_result.feature_names,
        output_path=feature_schema_path,
    )

    create_prediction_plot(
        y_true=y_test,
        y_pred=y_pred,
        output_path=plot_path,
    )

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "project": "dog-aging-project-mlops",
                "task": "cslb-score-regression",
                "data_source": "DAP 2021 curated data",
                "model_family": training_result.model_name,
            }
        )

        mlflow.log_params(
            {
                "model_name": training_result.model_name,
                "test_size": test_size,
                "random_state": random_state,
                "dataset_path": str(dataset_path),
                "feature_count": len(training_result.feature_names),
                **{
                    f"model__{key}": value
                    for key, value in training_result.model_params.items()
                    if isinstance(value, (str, int, float, bool))
                    or value is None
                },
            }
        )

        mlflow.log_metrics(training_result.metrics)

        mlflow.log_artifact(feature_schema_path)

        mlflow.log_artifact(plot_path)

        mlflow.sklearn.log_model(
            sk_model=training_result.model,
            artifact_path="model",
            input_example=pd.DataFrame(
                [dict.fromkeys(training_result.feature_names, 0.0)]
            ),
        )

        return run.info.run_id