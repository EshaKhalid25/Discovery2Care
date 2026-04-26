import json
import os
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[2]
# Streamlit may start with a different cwd; always load project .env first.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    # Guard against accidental "KEY=VALUE" pasted into value field.
    prefix = f"{name}="
    if value.startswith(prefix):
        value = value[len(prefix):].strip()
    return value


def databricks_status() -> dict[str, Any]:
    host = _env("DATABRICKS_HOST")
    token = _env("DATABRICKS_TOKEN")
    vector_index = _env("VECTOR_SEARCH_INDEX")
    llm_endpoint = _env("DATABRICKS_LLM_ENDPOINT")
    groq_key = _env("GROQ_API_KEY")
    return {
        "host_configured": bool(host),
        "token_configured": bool(token),
        "vector_index_configured": bool(vector_index),
        "llm_endpoint_configured": bool(llm_endpoint),
        "groq_configured": bool(groq_key),
        "mlflow_experiment_configured": bool(_env("MLFLOW_EXPERIMENT_PATH")),
        "is_ready": bool(host and token and vector_index),
    }


def _api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    host = _env("DATABRICKS_HOST")
    token = _env("DATABRICKS_TOKEN")
    if not host or not token:
        raise RuntimeError("Databricks host/token are not configured.")

    base = host.rstrip("/")
    url = f"{base}{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, data=data, method=method.upper())
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Databricks API error {exc.code}: {body[:500]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Databricks connection error: {exc}") from exc


def query_vector_search(
    query_text: str,
    num_results: int = 12,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    index_name = _env("VECTOR_SEARCH_INDEX")
    if not index_name:
        return []

    encoded_index = parse.quote(index_name, safe="")
    path = f"/api/2.0/vector-search/indexes/{encoded_index}/query"
    payload: dict[str, Any] = {
        "query_text": query_text,
        "num_results": num_results,
    }
    if columns:
        payload["columns"] = columns

    response = _api_request("POST", path, payload)
    result = response.get("result", {})
    data_array = result.get("data_array", [])
    manifest = result.get("manifest", {})
    cols = [c.get("name") for c in manifest.get("columns", []) if c.get("name")]

    if not data_array:
        return []
    if not cols:
        # Fallback if manifest is absent
        return [{"raw": row} for row in data_array]

    rows: list[dict[str, Any]] = []
    for row in data_array:
        item = {}
        for idx, col in enumerate(cols):
            item[col] = row[idx] if idx < len(row) else None
        rows.append(item)
    return rows


def _compact_candidates_for_summary(candidates: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    """Normalize vector-search rows and local ranked rows into one prompt shape."""
    compact: list[dict[str, Any]] = []
    for c in candidates[:limit]:
        name = str(c.get("name") or c.get("facility") or "Unknown").strip()
        city = str(c.get("address_city") or "").strip()
        state = str(c.get("address_state_or_region_clean") or "").strip()
        loc = str(c.get("location") or "").strip()
        if loc and (not city or not state):
            parts = [p.strip() for p in loc.split(",") if p.strip()]
            if len(parts) >= 2:
                city = city or parts[0]
                state = state or parts[-1]
            elif len(parts) == 1 and not city:
                city = parts[0]
        desc = str(c.get("description", "") or "").strip()[:220]
        if not desc:
            ev = c.get("evidence")
            if isinstance(ev, list) and ev:
                desc = " | ".join(str(x) for x in ev[:2])[:220]
        compact.append({"name": name, "city": city, "state": state, "signals": desc})
    return compact


def _build_summary_prompt(query: str, compact: list[dict[str, Any]]) -> str:
    return (
        "You help users interpret a STATIC facility directory for India. You are not a doctor.\n\n"
        "Write 4-7 short sentences in clear English. Follow ALL of these rules:\n"
        "1) Use ONLY the candidate records below. If none clearly match the clinical need "
        "(e.g. user asks kidney/urology/emergency urine symptoms but no urology/nephrology signals appear), "
        "say that honestly — do not pretend a match exists.\n"
        "2) Kidney stone or inability to pass urine often needs urgent in-person assessment. "
        "Do NOT assume dialysis is required; dialysis is different from acute stone or retention unless "
        "the records explicitly mention dialysis or chronic kidney failure needing it.\n"
        "3) Say clearly that this list may be incomplete or wrong for true emergencies: people should use "
        "local emergency services / nearest ER / call ahead — not rely on this app alone.\n"
        "4) If the directory is a weak fit for an emergency, say that picking a name from incomplete data "
        "can be misleading and they should seek immediate care.\n"
        "5) Never invent phone numbers, doctors, bed availability, or services.\n\n"
        f"User query: {query}\n"
        f"Candidate facilities (excerpts only): {json.dumps(compact)}"
    )


def _call_databricks_llm(prompt: str) -> str:
    endpoint = _env("DATABRICKS_LLM_ENDPOINT")
    if not endpoint:
        return ""
    path = f"/serving-endpoints/{endpoint}/invocations"
    payload = {
        "messages": [
            {"role": "system", "content": "You write concise, grounded healthcare routing rationale."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 220,
        "temperature": 0.15,
    }
    response = _api_request("POST", path, payload)

    if isinstance(response, dict):
        if "choices" in response and response["choices"]:
            msg = response["choices"][0].get("message", {})
            if isinstance(msg, dict):
                return str(msg.get("content", "")).strip()
        preds = response.get("predictions")
        if isinstance(preds, list) and preds:
            first = preds[0]
            if isinstance(first, dict):
                if "content" in first:
                    return str(first["content"]).strip()
                if "text" in first:
                    return str(first["text"]).strip()
            return str(first).strip()
    return ""


def _call_groq_chat(prompt: str) -> str:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        return ""
    model = _env("GROQ_MODEL", "llama-3.1-8b-instant")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise, grounded healthcare routing text. "
                    "You are not a doctor; encourage urgent care when symptoms sound severe."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 280,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    # Groq sits behind Cloudflare; requests without User-Agent often get 403 error 1010.
    req.add_header(
        "User-Agent",
        "Discovery2Care/1.0 (Streamlit; +https://github.com/) Python-urllib",
    )
    try:
        with request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            body = json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq API error {exc.code}: {err_body[:400]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Groq connection error: {exc}") from exc

    choices = body.get("choices") if isinstance(body, dict) else None
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            return str(msg.get("content", "")).strip()
    return ""


def call_ai_summary(query: str, candidates: list[dict[str, Any]]) -> tuple[str, str, str]:
    """
    Generate a short natural-language summary.
    Prefers Groq when GROQ_API_KEY is set (reliable for local demos), else Databricks serving.
    Returns (summary_text, provider_label, error_hint).
    provider is \"Groq\", \"Databricks\", or \"\". error_hint is non-empty if LLM was expected but failed.
    """
    compact = _compact_candidates_for_summary(candidates)
    if not compact:
        return "", "", ""

    prompt = _build_summary_prompt(query, compact)
    wants_llm = bool(_env("GROQ_API_KEY") or _env("DATABRICKS_LLM_ENDPOINT"))
    last_err = ""

    # Groq first when configured — works when Databricks model serving is unavailable.
    if _env("GROQ_API_KEY"):
        try:
            text = _call_groq_chat(prompt)
            if text:
                return text, "Groq", ""
        except Exception as exc:
            last_err = str(exc)[:280]

    if _env("DATABRICKS_LLM_ENDPOINT"):
        try:
            text = _call_databricks_llm(prompt)
            if text:
                return text, "Databricks", ""
        except Exception as exc:
            last_err = str(exc)[:280]

    if wants_llm and last_err:
        return "", "", last_err
    return "", "", last_err


def call_llm_summary(query: str, candidates: list[dict[str, Any]]) -> str:
    text, _, _ = call_ai_summary(query, candidates)
    return text
