from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dap.data.loader import CSLB_FEATURES


TARGET_COLUMN = "cslb_score"


@dataclass
class CSLBTrainingResult:
    model: object
    feature_names: list[str]
    metrics: dict[str, float]
    model_name: str
    model_params: dict
    test_target: pd.Series
    test_predictions: np.ndarray


def validate_model_data(data: pd.DataFrame) -> None:
    required_columns = set(CSLB_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing model columns: "
            f"{sorted(missing_columns)}"
        )


def build_model(
    model_name: str,
    random_state: int,
    alpha: float = 1.0,
    n_estimators: int = 200,
    max_depth: int | None = None,
):
    if model_name == "linear_regression":
        return LinearRegression()

    if model_name == "ridge":
        return Ridge(alpha=alpha)

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        "Unsupported model_name. Choose one of: "
        "'linear_regression', 'ridge', 'random_forest'."
    )


def validate_prediction_input(
    input_data: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    missing_features = set(feature_names) - set(input_data.columns)

    if missing_features:
        raise ValueError(
            "Input is missing required prediction features: "
            f"{sorted(missing_features)}"
        )

    ordered_input = input_data[feature_names].copy()

    non_numeric_columns = ordered_input.select_dtypes(
        exclude=["number", "bool"],
    ).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            "Prediction features must be numeric. Invalid columns: "
            f"{non_numeric_columns}"
        )

    if ordered_input.isna().any().any():
        missing_value_columns = ordered_input.columns[
            ordered_input.isna().any()
        ].tolist()

        raise ValueError(
            "Prediction input contains missing values in: "
            f"{missing_value_columns}"
        )

    return ordered_input


def train_cslb_model(
    data: pd.DataFrame,
    model_name: str = "linear_regression",
    test_size: float = 0.2,
    random_state: int = 42,
    alpha: float = 1.0,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> CSLBTrainingResult:
    validate_model_data(data)

    features = data[CSLB_FEATURES].copy()
    target = data[TARGET_COLUMN].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model(
        model_name=model_name,
        random_state=random_state,
        alpha=alpha,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    model_params = model.get_params()

    return CSLBTrainingResult(
        model=model,
        feature_names=CSLB_FEATURES,
        metrics=metrics,
        model_name=model_name,
        model_params=model_params,
        test_target=y_test,
        test_predictions=predictions,
    )


def predict_cslb(
    trained_model: CSLBTrainingResult,
    input_data: pd.DataFrame,
) -> np.ndarray:
    validated_input = validate_prediction_input(
        input_data=input_data,
        feature_names=trained_model.feature_names,
    )

    return trained_model.model.predict(validated_input)