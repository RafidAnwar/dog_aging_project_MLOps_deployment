# Local data directory

This folder already contains the preprocessed ML ready final.csv dataset. So no need to create it again, but the process for creation is:

1. Obtain approved access to Dog Aging Project Curated Data and required datasets.
2. Run the sql query in:

```text
sql/create_final_dataset.sql
```

3. Export its result as `final.csv`.
4. Place the file in this folder.

## Data Use Agreement

Do not upload DAP raw data or generated data to a public repository or for commercial usage.