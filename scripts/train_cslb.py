import json
import logging

import joblib

from dap.config import DATA_PATH, MODEL_PATH
from dap.data.loader import load_data
from dap.models.cslb_model import train_cslb_model
from dap.utils.logging_config import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Loading prepared DAP data from %s", DATA_PATH)
    data, _ = load_data(DATA_PATH)

    logger.info("Training CSLB linear regression model")
    training_result = train_cslb_model(
        data=data,
        test_size=0.2,
        random_state=42,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_result, MODEL_PATH)

    metrics_path = MODEL_PATH.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(training_result.metrics, indent=2),
        encoding="utf-8",
    )

    logger.info("Model artifact saved to %s", MODEL_PATH)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info("Evaluation metrics: %s", training_result.metrics)


if __name__ == "__main__":
    main()