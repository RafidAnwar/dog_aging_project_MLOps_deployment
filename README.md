# 🐕 DAP by Analytic Avengers >> ![Continuous Integration](https://github.com/RafidAnwar/dog_aging_project_MLOps_deployment/actions/workflows/ci.yml/badge.svg) ![Continuous Deployment](https://github.com/RafidAnwar/dog_aging_project_MLOps_deployment/actions/workflows/deploy.yml/badge.svg)

A web app that predicts canine cognitive dysfunction and analyses disease risk factors using 40,000+ samples from the [Dog Aging Project](https://dogagingproject.org).

Dog owners often struggle to detect early signs of cognitive dysfunction in aging pets or understand how specific factors contribute to common diseases, lacking simple tools for proactive care.
This project addresses these challenges through two core achievements: (1) delivering personalized cognitive dysfunction predictions via interactive questionnaires that track health changes over time (including 6-month follow-ups), and (2) providing tailored regression-based recommendations of the user’s dog breed to mitigate nine key physical diseases by analyzing user-selected variables. Leveraging over 40,000 samples from Dog Aging Project.org, it transforms research data into an accessible app that empowers owners with actionable insights for better canine health outcomes.

---

## What It Does

**Cognitive Dysfunction Prediction** — answer a short questionnaire about your dog and get an instant cognitive health score. If the result is borderline, a 6-month follow-up assessment is unlocked to track changes over time.

**Disease Regression Analysis** — select your dog breed and find out which disease is most common for your dog. Then select any of 9 diseases (cancer, cardiac, kidney, etc.) and explore how diet, physical activity, behaviour, or environment statistically affects disease odds and get personalized suggestions. 


## Architecture

```text
Approved DAP data access
        |
        v
sql/create_final_dataset.sql
        |
        v
data/final.csv
        |
        +----------------------------+
        |                            |
        |                            v
        |                   scripts/train_cslb.py
        |                            |
        |                            v
        |                        MLflow tracking
        |                            |
        v                            v
     app.py  <------------artifacts/cslb_model.joblib
        |
        v
Docker image
        |
        v
Docker Hub
        |
        v
GitHub Actions CI/CD
        |
        v
Azure App Service
        |
        v
Streamlit Webapp interface
```

## Local installation

```bash
git clone https://github.com/RafidAnwar/dog_aging_project_MLOps_deployment.git
cd dog_aging_project_MLOps_deployment

python -m venv .venv
```

Install the project and developer tools:

```bash
pip install -e ".[dev]"
```
## Data setup

This repository already contains the project-ready final.csv used by webapp.

Steps to regenerate Data from DAP Source (Requires DAP Access)
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

## Experiment tracking with MLflow

This project tracks CSLB regression experiments locally using MLflow.

### Train a baseline

```bash
python scripts/train_cslb.py --model-name linear_regression
```

### Compare models

```bash
python scripts/train_cslb.py --model-name ridge --alpha 0.1
python scripts/train_cslb.py --model-name ridge --alpha 1.0
python scripts/train_cslb.py --model-name random_forest --n-estimators 200
```

### Open the MLflow UI

```bash
mlflow server 
```

Then open `http://127.0.0.1:5000`.

Each training run logs:

- Model type and hyperparameters.
- MAE, RMSE, and R².
- Feature schema.
- Prediction-error plot.
- Serialized scikit-learn model artifact.

The Streamlit interface loads the model saved at:

```text
artifacts/cslb_model.joblib
```

After choosing a preferred run, rerun that configuration to make it the
local serving artifact.

## Test the project

```bash
pytest
ruff check .
```

## Run the Streamlit app

```bash
streamlit run app.py
```
## Containarization via Docker

Build the application image:

```bash
docker build -t dap-mlops .
```

Run the container locally:

```bash
docker run --rm -p 8501:8501 dap-mlops
```

Then Open:

http://localhost:8501

## CI/CD and Deployment

This project uses GitHub Actions to automate quality checks, Docker image creation, Docker Hub publishing, and deployment to Azure App Service.

### CI workflow

The CI workflow runs automatically on every push to `main` and on pull requests targeting `main`.

It performs the following checks:

- Runs `ruff check .` for code-quality and linting checks.
- Runs `pytest` for automated tests.
- Builds the Docker image to verify that the application can be containerized.

### CD workflow

The CD workflow runs when changes are merged or pushed to the `main` branch.

It performs the following steps:

1. Builds a new Docker image from the latest `main` branch.
2. Pushes the image to Docker Hub.
3. Restarts the Azure App Service.
4. Azure pulls the updated image and serves the updated application.


### Deployment pipeline

```text
Push or merge to main
        |
        v
GitHub Actions
        |
        +--> Ruff linting
        |
        +--> Pytest tests
        |
        v
Docker image build
        |
        v
Push to Docker Hub
        |
        v
Azure App Service restart
        |
        v
Updated Streamlit web application
```

### Docker Hub image

The deployed Docker image is available at:

```text
[https://hub.docker.com/r/rafidanwar/dap](https://hub.docker.com/r/rafidanwar/dap)
```

## Webapp Usage Guide
- Select the analysis you want to run.
  
**Cognitive dysfunction prediction:**
- Complete all questionnaires about your dog. Depending on the dogs condition, you may need observe for 6 months and then answer 6 more followup questions.
  
**Disease regression analysis:**
- Choose your dog breed.
- Select a disease and choose either one or more variables.

---
## Repository layout

```text
sql/                   Reproducible SQL dataset-selection query
data/                  Processed dataset used by the application
src/dap/               Reusable Python application package
scripts/               Repeatable command-line workflows
tests/                 Automated tests using synthetic data
artifacts/             Trained model artifacts used by the application
.github/workflows/     GitHub Actions CI/CD workflows
Dockerfile             Docker configuration for the Streamlit application
```

## Tech Stack

Python · Streamlit · SQL · MLflow · Docker · GitHub Actions · Docker Hub · Azure App Service 

<img width="975" height="374" alt="image" src="https://github.com/user-attachments/assets/6ec24864-ced5-47e9-b0e0-a5ee23516efa" />

<img width="975" height="865" alt="image" src="https://github.com/user-attachments/assets/b3041261-55e8-4ec3-a07e-756b4dbb84bd" />


> Educational project only. This application is not a veterinary diagnostic
> device and must not be used for clinical or treatment decisions.