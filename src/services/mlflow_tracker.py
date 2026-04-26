import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def _get_mlflow():
    try:
        import mlflow  # type: ignore
        return mlflow
    except Exception:
        return None


def _configure_tracking(mlflow) -> None:
    host = os.getenv("DATABRICKS_HOST", "").strip()
    token = os.getenv("DATABRICKS_TOKEN", "").strip()
    exp = os.getenv("MLFLOW_EXPERIMENT_PATH", "").strip()

    if host and token:
        os.environ.setdefault("DATABRICKS_HOST", host)
        os.environ.setdefault("DATABRICKS_TOKEN", token)
        mlflow.set_tracking_uri("databricks")
        if exp:
            try:
                mlflow.set_experiment(exp)
            except Exception:
                pass
    else:
        mlflow.set_tracking_uri("mlruns")


def log_agent_query_run(query: str, result: dict[str, Any]) -> str:
    """
    Log one agent query run to MLflow and return run_id.
    Returns "" if MLflow is unavailable/fails.
    """
    mlflow = _get_mlflow()
    if mlflow is None:
        return ""

    try:
        _configure_tracking(mlflow)
        with mlflow.start_run(run_name="agent_chat_query"):
            mlflow.log_param("query", query[:500])
            mlflow.log_param("engine", str(result.get("engine", "Local")))
            mlflow.log_param("need", str(result.get("need", "general")))
            mlflow.log_param("fallback_mode", str(result.get("fallback_mode", "strict")))
            mlflow.log_param("llm_provider", str(result.get("llm_provider", "")))
            err = str(result.get("llm_error", "")).strip()
            if err:
                mlflow.log_param("llm_error", err[:500])
            mlflow.log_param("states_detected", len(result.get("states", [])))
            mlflow.log_param("cities_detected", len(result.get("cities", [])))

            items = result.get("results", [])
            mlflow.log_metric("results_count", len(items))
            if items:
                avg_match = sum(int(i.get("match_score", 0)) for i in items) / len(items)
                avg_trust = sum(int(i.get("trust_score", 0)) for i in items) / len(items)
                mlflow.log_metric("avg_match_score", avg_match)
                mlflow.log_metric("avg_trust_score", avg_trust)

            llm_summary = str(result.get("llm_summary", "")).strip()
            if llm_summary:
                mlflow.log_text(llm_summary, "summary.txt")

            # Save compact result payload as artifact for traceability
            compact_rows = []
            for i in items[:10]:
                compact_rows.append(
                    {
                        "facility": i.get("facility"),
                        "location": i.get("location"),
                        "match_score": i.get("match_score"),
                        "trust_score": i.get("trust_score"),
                        "trust_label": i.get("trust_label"),
                        "evidence": i.get("evidence", [])[:2],
                    }
                )
            mlflow.log_dict(
                {
                    "query": query,
                    "need": result.get("need"),
                    "engine": result.get("engine"),
                    "fallback_mode": result.get("fallback_mode"),
                    "keywords": result.get("keywords", []),
                    "results": compact_rows,
                },
                "agent_result.json",
            )

            run = mlflow.active_run()
            return run.info.run_id if run else ""
    except Exception:
        return ""
