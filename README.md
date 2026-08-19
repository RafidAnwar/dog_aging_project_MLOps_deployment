# 🐕 DAP by Analytic Avengers

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
data/final.csv  (created local data)
        |
        +----------------------------+
        |                            |
        |                            v
        |                   scripts/train_cslb.py
        |                            |
        v                            v
     app.py  <------------artifacts/cslb_model.joblib
        |
        v
Streamlit Webapp interface
```

## Data setup

This repository does not include DAP data due to data user agreement.

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

Install the project and developer tools:

```bash
pip install -e ".[dev]"
```

## Test the project

```bash
pytest
ruff check .
```

## Train the model

After the final data`final.csv` is placed in`data/`:

```bash
python scripts/train_cslb.py
```

This creates local artifacts of the CSLB model:

```text
artifacts/cslb_model.joblib
artifacts/cslb_model.metrics.json
```

## Run the Streamlit app

```bash
streamlit run app.py
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
sql/        Reproducible SQL dataset-selection query
data/       Local final.csv only; excluded from Git
src/dap/    Reusable Python application package
scripts/    Repeatable command-line workflows
tests/      Automated tests using synthetic data
artifacts/  Local trained models and metrics
```

## Tech Stack

Python · Streamlit · SQL · GitHub

<img width="975" height="374" alt="image" src="https://github.com/user-attachments/assets/6ec24864-ced5-47e9-b0e0-a5ee23516efa" />
---
<img width="975" height="865" alt="image" src="https://github.com/user-attachments/assets/b3041261-55e8-4ec3-a07e-756b4dbb84bd" />
---

> Educational project only. This application is not a veterinary diagnostic
> device and must not be used for clinical or treatment decisions.