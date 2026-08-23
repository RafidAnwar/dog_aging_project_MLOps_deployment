import logging
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import yaml

from dap.config import DATA_PATH, MODEL_PATH
from dap.data.loader import load_data
from dap.models.cslb_model import train_cslb_model
from dap.tracking.mlflow_tracker import (
    configure_mlflow,
    log_training_run,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Get current Git commit hash for reproducibility."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def load_params(params_path: Path = Path("params.yaml")) -> dict:
    """Load parameters from params.yaml."""
    if not params_path.exists():
        logger.warning("params.yaml not found, using defaults")
        return {
            "train": {
                "model_name": "linear_regression",
                "test_size": 0.2,
                "random_state": 42,
            }
        }

    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main training pipeline."""
    # Load parameters from params.yaml (managed by DVC)
    params = load_params()
    train_params = params.get("train", {})

    model_name = train_params.get("model_name", "linearregression")
    test_size = train_params.get("test_size", 0.2)
    random_state = train_params.get("random_state", 42)

    logger.info("Loading data from %s", DATA_PATH)
    data, _ = load_data(DATA_PATH)

    logger.info("Configuring MLflow")
    configure_mlflow()

    logger.info("Training %s model", model_name)
    logger.info("Parameters: test_size=%s, random_state=%s", test_size, random_state)

    training_result = train_cslb_model(
        data=data,
        modelname=model_name,
        testsize=test_size,
        randomstate=random_state,
    )

    # Get Git commit for DVC integration
    git_commit = get_git_commit_hash()

    tracking_output_dir = Path("artifacts") / "mlflow"
    run_id = log_training_run(
        training_result=training_result,
        dataset_path=DATA_PATH,
        test_size=test_size,
        random_state=random_state,
        output_dir=tracking_output_dir,
        ytest=training_result.testtarget,
        ypred=training_result.testpredictions,
        git_commit=git_commit,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(training_result, MODEL_PATH)

    logger.info("Model artifact saved to %s", MODEL_PATH)
    logger.info("MLflow run ID: %s", run_id)
    logger.info("Git commit: %s", git_commit)
    logger.info("Metrics: %s", training_result.metrics)


if __name__ == "__main__":
    main()