# Discovery2Care

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run data cleaning

```bash
python scripts/clean_data.py
```

## Run frontend (Streamlit)

```bash
streamlit run app.py
```

## Files

- Raw data: `data/raw/facilities_raw.csv`
- Cleaned data: `data/processed/facilities_clean.csv`
- Cleaning report: `data/processed/cleaning_report.json`

## Optional Databricks integration (Agent Chat)

Set these environment variables to enable Databricks-powered retrieval and LLM summary:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `VECTOR_SEARCH_INDEX`
- `DATABRICKS_LLM_ENDPOINT` (optional, for LLM summary)
- `MLFLOW_EXPERIMENT_PATH` (optional, for Databricks MLflow tracking)
