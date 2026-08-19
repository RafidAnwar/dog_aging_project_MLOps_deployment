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