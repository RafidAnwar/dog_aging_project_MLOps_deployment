import pytest

from dap.models.cslb_model import build_model


def test_builds_linear_regression_model():
    model = build_model(
        model_name="linear_regression",
        random_state=42,
    )

    assert model.__class__.__name__ == "LinearRegression"


def test_builds_ridge_model():
    model = build_model(
        model_name="ridge",
        random_state=42,
        alpha=0.5,
    )

    assert model.__class__.__name__ == "Ridge"
    assert model.alpha == 0.5


def test_builds_random_forest_model():
    model = build_model(
        model_name="random_forest",
        random_state=42,
        n_estimators=50,
        max_depth=5,
    )

    assert model.__class__.__name__ == "RandomForestRegressor"
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported"):
        build_model(
            model_name="unknown",
            random_state=42,
        )