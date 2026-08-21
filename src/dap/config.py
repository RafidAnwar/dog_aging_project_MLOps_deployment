import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path(
    os.getenv(
        "DAP_DATA_PATH",
        PROJECT_ROOT / "data" / "final.csv",
    )
)

MODEL_PATH = Path(
    os.getenv(
        "CSLB_MODEL_PATH",
        PROJECT_ROOT / "artifacts" / "cslb_model.joblib",
    )
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "dap-cslb-regression",
)

MLFLOW_ARTIFACT_ROOT = Path(
    os.getenv(
        "MLFLOW_ARTIFACT_ROOT",
        PROJECT_ROOT / "mlartifacts",
    )
)