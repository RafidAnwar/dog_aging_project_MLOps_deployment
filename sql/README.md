# Create final dataset:

This repository does not contain Dog Aging Project (DAP) data due to Data Use Agreement. 
Use this create_final_dataset.sql to create the project-ready final.csv used by webapp.

## Required DAP source tables

The SQL query requires access to:

- `DAP_2021_HLES_dog_owner_v1`
- `DAP_2021_CSLB_v1`
- `DAP_2021_DogOverview_v1`

## Steps

1. Obtain approved access to Dog Aging Project Curated Data.
2. Open the SQL environment where the required tables are available.
3. Run `create_final_dataset.sql`.
4. Export the query result as CSV.
5. Name the file exactly: 'final.csv'
6. Place it in:

```text
data/final.csv
```

## Output validation

The generated CSV must include at least:

- `dog_id`
- `Breed`
- `cslb_score`
- `dd_age_years`
- `dd_weight_lbs`
- the twelve CSLB model feature columns

## Data-use rule

Do not commit or upload DAP raw data or the generated `final.csv` to a public GitHub repository.