"""
================================================================================
Discovery2Care — Databricks Unity Catalog + Delta setup (Vector Search prep)
================================================================================

WHO SHOULD RUN THIS?
  Someone with access to your Databricks workspace (notebook or job), NOT your
  laptop-only Python venv. This file uses PySpark (`spark`), which Databricks
  provides automatically.

WHAT IT DOES (short)
  1) Reads your *cleaned* facilities CSV from DBFS or a Unity Catalog Volume.
  2) Creates/overwrites a Delta table:   catalog.schema.<base_table>
  3) Creates/overwrites a Delta table:   catalog.schema.<vs_table>
     with an extra column `combined_text` (text for embeddings / retrieval).
  4) Creates/overwrites a *slim* Delta table: catalog.schema.<vs_slim_table>
     with only ~11 columns — required because Mosaic Vector Search caps index
     fields (e.g. 50). Point your Vector Search *source* at the SLIM table.

BEFORE YOU RUN
  A) On your PC, generate the cleaned file:
         python scripts/clean_data.py
     Output: data/processed/facilities_clean.csv

  B) Upload that CSV to Databricks, e.g.:
       - Workspace → Upload to DBFS: dbfs:/FileStore/discovery2care/facilities_clean.csv
       - Or put it on a Volume:
         /Volumes/<catalog>/<schema>/<volume>/facilities_clean.csv

  C) In Databricks, attach this notebook to a cluster or use serverless SQL
     / notebook where `spark` is defined.

HOW TO RUN (Databricks notebook — recommended)
  Option 1 — copy this whole file into a notebook cell (or repo file), then:

      from scripts.setup_databricks_tables import run_setup
      run_setup(source_csv_path="dbfs:/FileStore/discovery2care/facilities_clean.csv")

  Option 2 — if the file lives in Repos:

      %run ./scripts/setup_databricks_tables

      run_setup("dbfs:/FileStore/discovery2care/facilities_clean.csv")

  Adjust `source_csv_path` to YOUR path.

AFTER THIS SCRIPT
  • In Databricks UI: Mosaic AI → Vector Search → create/sync an index whose
    *source* is the SLIM table (e.g. ...india_healthcare_facilities_vs_slim).
  • Set primary key column to `row_id` and embedding text column to
    `combined_text`.
  • Put the full index name in your app `.env` as VECTOR_SEARCH_INDEX.

LOCAL PC
  Running `python scripts/setup_databricks_tables.py` on Windows/Mac without
  Spark will fail by design — use Databricks.

================================================================================
"""

from __future__ import annotations

from typing import Optional


# Defaults aligned with this repo’s Unity Catalog layout (change if yours differs).
DEFAULT_CATALOG = "multiagenthealthcare"
DEFAULT_SCHEMA = "discovery2care"
DEFAULT_BASE_TABLE = "india_healthcare_facilities"
DEFAULT_VS_TABLE = "india_healthcare_facilities_vs"
DEFAULT_VS_SLIM_TABLE = "india_healthcare_facilities_vs_slim"


def _ensure_spark():
    """Return Spark session when running inside Databricks."""
    try:
        return spark  # noqa: F821  # injected by Databricks runtime
    except NameError as exc:  # pragma: no cover
        raise RuntimeError(
            "No `spark` session. Run this script from a Databricks notebook or job, "
            "not with plain `python` on your laptop."
        ) from exc


def _fq(catalog: str, schema: str, table: str) -> str:
    return f"`{catalog}`.`{schema}`.`{table}`"


def run_setup(
    source_csv_path: str,
    catalog: str = DEFAULT_CATALOG,
    schema: str = DEFAULT_SCHEMA,
    base_table: str = DEFAULT_BASE_TABLE,
    vs_table: str = DEFAULT_VS_TABLE,
    vs_slim_table: str = DEFAULT_VS_SLIM_TABLE,
    id_column: Optional[str] = None,
    skip_slim: bool = False,
) -> None:
    """
    Build Delta tables for Discovery2Care.

    Args:
        source_csv_path: DBFS or Volume path to facilities_clean.csv
        catalog / schema: Unity Catalog location
        base_table: Main wide facilities table
        vs_table: Wide table + combined_text (may exceed Vector Search column limits)
        vs_slim_table: Narrow table for Vector Search source (keep field count low)
        id_column: Optional existing unique id column; else `row_id` is generated
        skip_slim: If True, only create base + vs_table (not recommended for VS UI limits)
    """
    spark_session = _ensure_spark()
    base_fqn = _fq(catalog, schema, base_table)
    vs_fqn = _fq(catalog, schema, vs_table)
    slim_fqn = _fq(catalog, schema, vs_slim_table)

    print("== Discovery2Care Databricks setup ==")
    print(f"Source CSV:        {source_csv_path}")
    print(f"Base table:        {base_fqn}")
    print(f"VS prep table:     {vs_fqn}")
    if not skip_slim:
        print(f"VS slim table:     {slim_fqn}  (use this as Vector Search *source*)")

    spark_session.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
    spark_session.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

    df = (
        spark_session.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(source_csv_path)
    )

    final_id_col = id_column
    if final_id_col is None or final_id_col not in df.columns:
        from pyspark.sql.functions import monotonically_increasing_id

        final_id_col = "row_id"
        if "row_id" in df.columns:
            df = df.drop("row_id")
        df = df.withColumn("row_id", monotonically_increasing_id().cast("string"))

    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(base_fqn)
    )

    spark_session.sql(
        f"""
        CREATE OR REPLACE TABLE {vs_fqn} AS
        SELECT
            *,
            concat_ws(' ',
                coalesce(cast(description as string), ''),
                coalesce(cast(capability as string), ''),
                coalesce(cast(procedure as string), ''),
                coalesce(cast(equipment as string), ''),
                coalesce(cast(specialties as string), '')
            ) AS combined_text
        FROM {base_fqn}
        """
    )

    if not skip_slim:
        # Keep column count under Vector Search index limits; index this table.
        spark_session.sql(
            f"""
            CREATE OR REPLACE TABLE {slim_fqn} AS
            SELECT
                row_id,
                name,
                address_city,
                address_state_or_region_clean,
                facility_type_id,
                cast(description as string) AS description,
                cast(specialties as string) AS specialties,
                cast(procedure as string) AS procedure,
                cast(equipment as string) AS equipment,
                cast(capability as string) AS capability,
                cast(core_fields_present as double) AS core_fields_present,
                combined_text
            FROM {vs_fqn}
            """
        )

    base_count = spark_session.sql(f"SELECT COUNT(*) AS c FROM {base_fqn}").collect()[0]["c"]
    vs_count = spark_session.sql(f"SELECT COUNT(*) AS c FROM {vs_fqn}").collect()[0]["c"]
    print("Setup complete.")
    print(f"- Base rows:  {base_count}")
    print(f"- VS rows:    {vs_count}")
    if not skip_slim:
        slim_count = spark_session.sql(f"SELECT COUNT(*) AS c FROM {slim_fqn}").collect()[0]["c"]
        slim_cols = len(spark_session.sql(f"DESCRIBE TABLE {slim_fqn}").collect())
        print(f"- Slim rows:  {slim_count}")
        print(f"- Slim columns (describe rows): {slim_cols}  → create Vector Search index on {slim_fqn}")
    print(f"- Primary key for index: {final_id_col}")
    print("- Embedding / text column: combined_text")
    print()
    print("Next: Databricks → Vector Search → Delta Sync index → source = SLIM table above.")
    print("      .env → VECTOR_SEARCH_INDEX=<catalog>.<schema>.<index_name> (API index name).")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Do not run this file with local Python.\n"
        "Open a Databricks notebook, upload facilities_clean.csv to DBFS/Volume, "
        "then call run_setup('dbfs:/.../facilities_clean.csv').\n"
        "Read the module docstring at the top of scripts/setup_databricks_tables.py for steps."
    )
