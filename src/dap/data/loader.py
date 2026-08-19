from pathlib import Path
import numpy as np
import pandas as pd


CSLB_FEATURES = [
    "dd_age_years",
    "dd_weight_lbs",
    "pa_activity_level",
    "hs_health_conditions_eye",
    "hs_health_conditions_ear",
    "cslb_pace",
    "cslb_stare",
    "cslb_stuck",
    "cslb_recognize",
    "cslb_walk_walls",
    "cslb_avoid",
    "cslb_find_food",
]

REQUIRED_COLUMNS = {
    "dog_id",
    "Breed",
    "cslb_score",
    *CSLB_FEATURES,
}


def load_data(data_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the project-ready `final.csv` created using the repository SQL file.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        A DataFrame indexed by dog_id and a sorted list of breeds.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n\n"
            "Create data/final.csv by following sql/README.md."
        )

    dataframe = pd.read_csv(path, low_memory=False)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns:\n"
            f"{sorted(missing_columns)}\n\n"
            "Regenerate the dataset using sql/create_final_dataset.sql."
        )

    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    data = dataframe.set_index("dog_id").copy()
    data = data.fillna(0)

    breeds = sorted(data["Breed"].dropna().unique().tolist())

    return data, breeds