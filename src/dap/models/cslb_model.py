from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dap.data.loader import CSLB_FEATURES


TARGET_COLUMN = "cslb_score"


@dataclass # A structured container
class CSLBTrainingResult:
    model: LinearRegression
    feature_names: list[str]
    metrics: dict[str, float]


def validate_model_data(data: pd.DataFrame) -> None:
    required_columns = set(CSLB_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing model columns: "
            f"{sorted(missing_columns)}"
        )


def train_cslb_model(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> CSLBTrainingResult:
    """
    Train and evaluate a Linear Regression model for CSLB score prediction.
    """
    validate_model_data(data)

    features = data[CSLB_FEATURES]
    target = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
    }

    return CSLBTrainingResult(
        model=model,
        feature_names=CSLB_FEATURES,
        metrics=metrics,
    )


def predict_cslb(
    trained_model: CSLBTrainingResult,
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Generate CSLB-score predictions for validated feature input."""
    missing_features = set(trained_model.feature_names) - set(input_data.columns)

    if missing_features:
        raise ValueError(
            f"Input is missing prediction features: {sorted(missing_features)}"
        )

    ordered_input = input_data[trained_model.feature_names]

    return trained_model.model.predict(ordered_input)