import pandas as pd
import pytest

from dap.data.loader import CSLB_FEATURES


@pytest.fixture
def sample_dap_dataframe() -> pd.DataFrame:
    rows = []

    for index in range(30):
        row = {
            "dog_id": index + 1,
            "Breed": "Labrador" if index % 2 == 0 else "Poodle",
            "cslb_score": float(index % 20),
            "hs_health_conditions_cancer": 0 if index % 2 == 0 else 2,
            "hs_health_conditions_skin": 0 if index % 3 == 0 else 2,
        }

        for feature_position, feature in enumerate(CSLB_FEATURES):
            row[feature] = float((index + feature_position) % 5)

        rows.append(row)

    return pd.DataFrame(rows)