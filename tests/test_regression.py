from dap.analysis.regression import get_most_common_disease_for_breed


def test_breed_summary_requires_breed_selection(sample_dap_dataframe):
    data = sample_dap_dataframe.set_index("dog_id")

    result = get_most_common_disease_for_breed(
        data,
        "— select a breed —",
    )

    assert "Select a breed" in result


def test_breed_summary_returns_text(sample_dap_dataframe):
    data = sample_dap_dataframe.set_index("dog_id")

    result = get_most_common_disease_for_breed(
        data,
        "Labrador",
    )

    assert "Labrador" in result