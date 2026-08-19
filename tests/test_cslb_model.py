from dap.data.loader import CSLB_FEATURES
from dap.models.cslb_model import predict_cslb, train_cslb_model


def test_train_cslb_model_returns_metrics(sample_dap_dataframe):
    data = sample_dap_dataframe.set_index("dog_id")

    training_result = train_cslb_model(data)

    assert "mae" in training_result.metrics
    assert "rmse" in training_result.metrics
    assert "r2" in training_result.metrics
    assert len(training_result.feature_names) == len(CSLB_FEATURES)


def test_predict_cslb_returns_one_prediction(sample_dap_dataframe):
    data = sample_dap_dataframe.set_index("dog_id")
    training_result = train_cslb_model(data)

    input_data = data[CSLB_FEATURES].iloc[[0]]
    prediction = predict_cslb(training_result, input_data)

    assert len(prediction) == 1