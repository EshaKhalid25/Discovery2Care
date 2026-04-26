import re
from typing import Any

import pandas as pd

from src.services.databricks_client import call_ai_summary, databricks_status, query_vector_search
from src.services.parsers import parse_list_cell


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "from",
    "near",
    "have",
    "has",
    "are",
    "can",
    "find",
    "show",
    "need",
    "where",
    "what",
    "which",
    "into",
    "about",
    "across",
    "within",
    "around",
    "state",
    "city",
    "area",
}

NEED_KEYWORDS = {
    "emergency": ["emergency", "trauma", "critical care", "appendectomy", "anesthesia", "surgery"],
    "oncology": ["oncology", "cancer", "tumor", "tumour", "chemotherapy", "radiation"],
    "dialysis": ["dialysis", "dialys", "nephrology", "renal", "kidney", "hemodialysis", "peritoneal"],
    "icu": ["icu", "intensive care", "critical care", "ventilator"],
    "neonatal": ["neonatal", "nicu", "newborn", "pediatric", "paediatric", "obstetrics"],
}


def _detect_need(query_lower: str) -> str:
    # Prioritize emergency/urology intent for urgent urinary-retention style queries.
    emergency_phrases = [
        "unable to pass urine",
        "cannot pass urine",
        "urinary retention",
        "kidney stone",
        "urgent",
        "urgently",
    ]
    if any(p in query_lower for p in emergency_phrases):
        return "emergency"

    for need, words in NEED_KEYWORDS.items():
        if any(w in query_lower for w in words):
            return need
    return "general"


def _extract_state_city_filters(df: pd.DataFrame, query_lower: str) -> tuple[list[str], list[str]]:
    states = [s for s in df["address_state_or_region_clean"].dropna().unique().tolist() if str(s).strip()]
    cities = [c for c in df["address_city"].dropna().unique().tolist() if str(c).strip()]

    def _phrase_in_query(phrase: str) -> bool:
        p = re.escape(phrase.strip().lower())
        if not p:
            return False
        return bool(re.search(rf"\b{p}\b", query_lower))

    # States: strict word-boundary phrase match
    matched_states = [s for s in states if _phrase_in_query(str(s))]

    # Cities: avoid noisy very short tokens that can match inside other words (e.g., "Una" in "unable")
    matched_cities = []
    for c in cities:
        city = str(c).strip()
        if len(city) < 4:
            continue
        if _phrase_in_query(city):
            matched_cities.append(c)

    # If a token appears in both state/city lists, treat it as city first unless state explicitly written.
    overlap = {str(x).lower() for x in matched_states} & {str(x).lower() for x in matched_cities}
    if overlap:
        matched_states = [s for s in matched_states if str(s).lower() not in overlap]
    return matched_states, matched_cities


def _build_search_blob(row: pd.Series) -> str:
    parts = [
        str(row.get("description", "")),
        " ".join(parse_list_cell(row.get("specialties"))),
        " ".join(parse_list_cell(row.get("procedure"))),
        " ".join(parse_list_cell(row.get("equipment"))),
        " ".join(parse_list_cell(row.get("capability"))),
    ]
    return " ".join(parts).lower()


def _extract_keywords(query_lower: str, need: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", query_lower)
    base = [w for w in words if w not in STOPWORDS]
    need_words = NEED_KEYWORDS.get(need, [])
    keywords = sorted(set(base + need_words))
    return keywords[:20]


def _compute_trust(row: pd.Series) -> tuple[int, str]:
    core_present = row.get("core_fields_present")
    core_present = 0 if pd.isna(core_present) else float(core_present)
    completeness = min(core_present / 5.0, 1.0)

    score = 60 + completeness * 30
    if bool(row.get("pin_invalid", False)):
        score -= 8
    if pd.isna(row.get("email_clean")) and not pd.isna(row.get("email")):
        score -= 4
    if pd.isna(row.get("official_phone_clean")) and not pd.isna(row.get("official_phone")):
        score -= 4

    score = int(max(0, min(100, round(score))))
    label = "High" if score >= 80 else ("Medium" if score >= 60 else "Low")
    return score, label


def _find_evidence(row: pd.Series, keywords: list[str]) -> list[str]:
    evidence_pool = parse_list_cell(row.get("capability")) + parse_list_cell(row.get("procedure"))
    if not evidence_pool:
        desc = str(row.get("description", "")).strip()
        return [desc[:220]] if desc else []

    matches = []
    for line in evidence_pool:
        low = str(line).lower()
        if any(k in low for k in keywords[:10]):
            matches.append(str(line))
        if len(matches) >= 2:
            break
    return matches if matches else [str(evidence_pool[0])]


def _find_evidence_from_dict(item: dict[str, Any], keywords: list[str]) -> list[str]:
    evidence_pool = (
        parse_list_cell(item.get("capability"))
        + parse_list_cell(item.get("procedure"))
        + parse_list_cell(item.get("equipment"))
    )
    if not evidence_pool:
        desc = str(item.get("description", "")).strip()
        return [desc[:220]] if desc else []
    matches = []
    for line in evidence_pool:
        low = str(line).lower()
        if any(k in low for k in keywords[:10]):
            matches.append(str(line))
        if len(matches) >= 2:
            break
    return matches if matches else [str(evidence_pool[0])]


def _score_candidates(
    candidate_df: pd.DataFrame,
    keywords: list[str],
    need: str,
    states: list[str],
    cities: list[str],
    strict_need: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in candidate_df.iterrows():
        blob = _build_search_blob(row)
        keyword_hits = sum(1 for k in keywords if k in blob)
        if strict_need and keyword_hits == 0 and need != "general":
            continue

        geo_bonus = 0
        if states and row.get("address_state_or_region_clean") in states:
            geo_bonus += 8
        if cities and row.get("address_city") in cities:
            geo_bonus += 10

        core_present = row.get("core_fields_present")
        core_present = 0 if pd.isna(core_present) else float(core_present)
        completeness_bonus = min(core_present / 5.0, 1.0) * 10

        # Fallback scoring still needs minimum signal so results are relevant.
        if not strict_need and keyword_hits == 0:
            keyword_hits = 1 if str(row.get("description", "")).strip() else 0

        match_score = int(min(99, round(keyword_hits * 9 + geo_bonus + completeness_bonus)))
        trust_score, trust_label = _compute_trust(row)
        evidence = _find_evidence(row, keywords)

        rows.append(
            {
                "facility": str(row.get("name", "Unknown Facility")),
                "location": f"{row.get('address_city', 'N/A')}, {row.get('address_state_or_region_clean', 'N/A')}",
                "match_score": match_score,
                "trust_score": trust_score,
                "trust_label": trust_label,
                "why": f"Matched {keyword_hits} relevant signals for this query.",
                "evidence": evidence,
            }
        )
    return sorted(rows, key=lambda x: (x["match_score"], x["trust_score"]), reverse=True)


def _local_summary(query: str, need: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return (
            "No strong exact matches found for this query in the current retrieval step. "
            "Try adding a nearby city/state or broader clinical terms."
        )

    top = rows[0]
    top_fac = top.get("facility", "a facility")
    top_loc = top.get("location", "selected area")
    if need == "emergency":
        return (
            f"For this urgent query, the best immediate match is {top_fac} in {top_loc}. "
            "Use the listed evidence and call the facility before travel to confirm emergency handling."
        )
    return (
        f"Top recommendation is {top_fac} in {top_loc}, based on capability signal match and trust scoring. "
        "Review evidence snippets for verification before final referral."
    )


def _score_vector_candidates(
    candidates: list[dict[str, Any]],
    keywords: list[str],
    need: str,
    states: list[str],
    cities: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        blob_parts = [
            str(item.get("description", "")),
            " ".join(parse_list_cell(item.get("specialties"))),
            " ".join(parse_list_cell(item.get("procedure"))),
            " ".join(parse_list_cell(item.get("equipment"))),
            " ".join(parse_list_cell(item.get("capability"))),
        ]
        blob = " ".join(blob_parts).lower()
        keyword_hits = sum(1 for k in keywords if k in blob)
        if keyword_hits == 0 and need != "general":
            continue

        geo_bonus = 0
        state_val = str(item.get("address_state_or_region_clean", ""))
        city_val = str(item.get("address_city", ""))
        if states and state_val in states:
            geo_bonus += 8
        if cities and city_val in cities:
            geo_bonus += 10

        core_present = item.get("core_fields_present", 3)
        try:
            core_present_f = float(core_present) if core_present is not None else 3.0
        except Exception:
            core_present_f = 3.0
        completeness_bonus = min(core_present_f / 5.0, 1.0) * 10

        match_score = int(min(99, round(keyword_hits * 9 + geo_bonus + completeness_bonus)))
        trust_score = int(min(100, max(0, round(55 + completeness_bonus * 2.5))))
        trust_label = "High" if trust_score >= 80 else ("Medium" if trust_score >= 60 else "Low")
        evidence = _find_evidence_from_dict(item, keywords)

        rows.append(
            {
                "facility": str(item.get("name") or item.get("facility") or "Unknown Facility"),
                "location": f"{city_val or 'N/A'}, {state_val or 'N/A'}",
                "match_score": match_score,
                "trust_score": trust_score,
                "trust_label": trust_label,
                "why": f"Matched {keyword_hits} relevant semantic signals from Databricks retrieval.",
                "evidence": evidence,
            }
        )

    return sorted(rows, key=lambda x: (x["match_score"], x["trust_score"]), reverse=True)


def run_agent_query(df: pd.DataFrame, query: str, top_k: int = 5) -> dict[str, Any]:
    query_lower = query.lower().strip()
    if not query_lower:
        return {
            "query": query,
            "need": "general",
            "states": [],
            "cities": [],
            "keywords": [],
            "fallback_mode": "strict",
            "engine": "Local",
            "llm_summary": "",
            "llm_provider": "",
            "llm_error": "",
            "results": [],
        }

    need = _detect_need(query_lower)
    states, cities = _extract_state_city_filters(df, query_lower)
    keywords = _extract_keywords(query_lower, need)

    # Primary path: Databricks retrieval + local scoring on retrieved rows
    dbx = databricks_status()
    if dbx["is_ready"]:
        try:
            vector_columns = [
                "name",
                "address_city",
                "address_state_or_region_clean",
                "facility_type_id",
                "description",
                "specialties",
                "procedure",
                "equipment",
                "capability",
                "core_fields_present",
            ]
            candidates = query_vector_search(query_text=query, num_results=max(12, top_k * 4), columns=vector_columns)
            if candidates:
                rows = _score_vector_candidates(candidates, keywords, need, states, cities)
                rows = rows[:top_k]
                ai_text, ai_provider, ai_err = call_ai_summary(query, candidates[:8])
                llm_summary = ai_text or _local_summary(query, need, rows)
                return {
                    "query": query,
                    "need": need,
                    "states": states,
                    "cities": cities,
                    "keywords": keywords[:10],
                    "fallback_mode": "databricks_primary",
                    "engine": "Databricks",
                    "llm_summary": llm_summary,
                    "llm_provider": ai_provider,
                    "llm_error": ai_err if not ai_text else "",
                    "results": rows,
                }
        except Exception:
            # Graceful fallback to local engine
            pass

    candidate_df = df.copy()
    if states:
        candidate_df = candidate_df[candidate_df["address_state_or_region_clean"].isin(states)]
    if cities:
        candidate_df = candidate_df[candidate_df["address_city"].isin(cities)]

    fallback_mode = "strict"
    if candidate_df.empty:
        # fallback 1: if filters are too narrow, use full dataset
        candidate_df = df.copy()
        fallback_mode = "relaxed_area"

    rows = _score_candidates(candidate_df, keywords, need, states, cities, strict_need=True)
    if not rows:
        # fallback 2: relax need strictness, keep area filter
        rows = _score_candidates(candidate_df, keywords, need, states, cities, strict_need=False)
        fallback_mode = "relaxed_need" if fallback_mode == "strict" else f"{fallback_mode}+relaxed_need"

    ranked = rows[:top_k]
    ai_text, ai_provider, ai_err = call_ai_summary(query, ranked)
    return {
        "query": query,
        "need": need,
        "states": states,
        "cities": cities,
        "keywords": keywords[:10],
        "fallback_mode": fallback_mode,
        "engine": "Local",
        "llm_summary": ai_text or _local_summary(query, need, ranked),
        "llm_provider": ai_provider,
        "llm_error": ai_err if not ai_text else "",
        "results": ranked,
    }
