import argparse
import logging
from pathlib import Path

import joblib

from dap.config import DATA_PATH, MODEL_PATH
from dap.data.loader import load_data
from dap.models.cslb_model import train_cslb_model
from dap.tracking.mlflow_tracker import (
    configure_mlflow,
    log_training_run,
)
from dap.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and track a CSLB regression model.",
    )

    parser.add_argument(
        "--model-name",
        choices=[
            "linear_regression",
            "ridge",
            "random_forest",
        ],
        default="linear_regression",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for random forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum random-forest tree depth.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    logger.info("Loading data from %s", DATA_PATH)
    data, _ = load_data(DATA_PATH)

    logger.info("Configuring MLflow")
    configure_mlflow()

    logger.info("Training %s", args.model_name)

    training_result = train_cslb_model(
        data=data,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
        alpha=args.alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    tracking_output_dir = Path("artifacts") / "tracking"

    run_id = log_training_run(
        training_result=training_result,
        dataset_path=DATA_PATH,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=tracking_output_dir,
        y_test=training_result.test_target,
        y_pred=training_result.test_predictions,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_result, MODEL_PATH)

    logger.info("Local serving artifact saved to %s", MODEL_PATH)
    logger.info("MLflow run ID: %s", run_id)
    logger.info("Metrics: %s", training_result.metrics)


if __name__ == "__main__":
    main()