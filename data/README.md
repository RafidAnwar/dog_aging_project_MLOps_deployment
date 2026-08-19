# Local data directory

This folder contains local data only. No Dog Aging Project data is included in this repository.

## Required file

Before running the project, create:

```text
data/final.csv
```

## How to create it

1. Obtain approved access to Dog Aging Project Curated Data.
2. Run the sql query in:

```text
sql/create_final_dataset.sql
```

3. Export its result as `final.csv`.
4. Place the file in this folder.

## Data Use Agreement

The `.gitignore` file excludes `data/*.csv`.

Do not remove that rule or upload DAP raw data or generated data to a public repository.