# Dog Aging Project MLOps Deployment

A production-oriented refactor of a Streamlit machine-learning application
built using Dog Aging Project (DAP) curated data.

> Educational project only. This application is not a veterinary diagnostic
> device and must not be used for clinical or treatment decisions.

## Features

- CSLB cognitive dysfunction score prediction using linear regression.
- Breed-level disease summaries.
- Logistic-regression association analysis across diet, activity, behavior,
  and environmental variables.
- Modular Python package structure.
- Reproducible model-training script.
- Automated tests using synthetic data.
- Foundation for future API, Docker, CI/CD, and cloud-deployment work.

## Architecture

```text
Approved DAP data access
        |
        v
sql/create_final_dataset.sql
        |
        v
data/final.csv  (local only; never committed)
        |
        +----------------------+
        |                      |
        v                      v
scripts/train_cslb.py       app.py
        |                      |
        v                      v
artifacts/cslb_model.joblib Streamlit interface
```

## Data setup

This repository does not include DAP data.

1. Obtain approved access to Dog Aging Project Curated Data.
2. Make these tables available in your SQL environment:
   - `DAP_2021_HLES_dog_owner_v1`
   - `DAP_2021_CSLB_v1`
   - `DAP_2021_DogOverview_v1`
3. Run:

```text
sql/create_final_dataset.sql
```

4. Export the result as:

```text
data/final.csv
```

See `sql/README.md` and `data/README.md` for details.

## Local installation

```bash
git clone https://github.com/RafidAnwar/dog_aging_project_MLOps_deployment.git
cd dog_aging_project_MLOps_deployment

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and developer tools:

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

## Test the project

```bash
pytest
ruff check .
```

## Train the model

After placing `final.csv` in `data/`:

```bash
python scripts/train_cslb.py
```

This creates local artifacts:

```text
artifacts/cslb_model.joblib
artifacts/cslb_model.metrics.json
```

## Run the Streamlit app

```bash
streamlit run app.py
```

## Repository layout

```text
sql/        Reproducible SQL dataset-selection query
data/       Local final.csv only; excluded from Git
src/dap/    Reusable Python application package
scripts/    Repeatable command-line workflows
tests/      Automated tests using synthetic data
artifacts/  Local trained models and metrics
```