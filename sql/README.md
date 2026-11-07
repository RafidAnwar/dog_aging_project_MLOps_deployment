# Create final dataset:

This repository contains the project-ready final.csv used by webapp.

It is created using this create_final_dataset.sql.

## Required DAP source tables

The SQL query requires access to these csv files:

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

Do not upload DAP raw data or the generated `final.csv` to another repository or use it for commercial purposes.