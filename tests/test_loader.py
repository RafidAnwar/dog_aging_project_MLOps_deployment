import pandas as pd
import pytest

from dap.data.loader import load_data


def test_load_data_returns_indexed_dataframe(
    tmp_path,
    sample_dap_dataframe,
):
    path = tmp_path / "final.csv"
    sample_dap_dataframe.to_csv(path, index=False)

    data, breeds = load_data(path)

    assert data.index.name == "dog_id"
    assert len(data) == 30
    assert breeds == ["Labrador", "Poodle"]


def test_load_data_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "not_found.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_load_data_raises_for_missing_columns(tmp_path):
    path = tmp_path / "final.csv"

    pd.DataFrame(
        {
            "dog_id": [1],
            "Breed": ["Labrador"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_data(path)