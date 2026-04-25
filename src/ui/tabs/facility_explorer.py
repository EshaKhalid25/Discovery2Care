import pandas as pd
import streamlit as st
import re

from src.services.parsers import parse_list_cell


DISPLAY_COLUMNS = {
    "name": "Name",
    "facility_type_id": "Facility Type ID",
    "address_city": "City",
    "address_state_or_region_clean": "State/Region",
    "pin_code_clean": "PIN Code",
    "official_phone_clean": "Official Phone",
    "core_fields_present": "Core Fields Present",
}


def _safe_text(value, fallback: str = "Not available") -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _humanize_text(text: str) -> str:
    val = text.strip()
    if not val:
        return val
    val = val.replace("_", " ").replace("-", " ")
    # Convert camelCase/PascalCase to spaced words
    val = re.sub(r"([a-z])([A-Z])", r"\1 \2", val)
    # Collapse extra whitespace
    val = re.sub(r"\s+", " ", val).strip()
    return val


def _render_structured_list(title: str, values: list[str]) -> None:
    clean_values = [_humanize_text(str(v)) for v in values if str(v).strip()]
    st.write(f"**{title}**")
    if not clean_values:
        st.write("Not available")
        return
    with st.expander(f"{title} ({len(clean_values)})", expanded=True):
        for idx, item in enumerate(clean_values, start=1):
            st.markdown(f"{idx}. {item}")


def _compute_trust_components(row: pd.Series) -> tuple[int, float, float, float]:
    # Completeness uses existing field from cleaning pipeline (0..5)
    core_present = row.get("core_fields_present")
    core_present = 0 if pd.isna(core_present) else float(core_present)
    completeness = min(core_present / 5.0, 1.0)

    # Consistency penalizes obvious quality flags
    consistency = 1.0
    if bool(row.get("pin_invalid", False)):
        consistency -= 0.2
    if pd.isna(row.get("email_clean")) and not pd.isna(row.get("email")):
        consistency -= 0.1
    if pd.isna(row.get("official_phone_clean")) and not pd.isna(row.get("official_phone")):
        consistency -= 0.1
    consistency = max(0.0, consistency)

    # Evidence strength based on non-empty extracted sections
    section_counts = 0
    non_empty_sections = 0
    for col in ["specialties", "procedure", "equipment", "capability"]:
        values = parse_list_cell(row.get(col))
        count = len([v for v in values if str(v).strip()])
        if count > 0:
            non_empty_sections += 1
            section_counts += min(count, 10)
    evidence_strength = min((section_counts / 30.0) * 0.7 + (non_empty_sections / 4.0) * 0.3, 1.0)

    trust_score = int(round((0.4 * completeness + 0.3 * consistency + 0.3 * evidence_strength) * 100))
    return trust_score, completeness, consistency, evidence_strength


def render_explorer_tab(df: pd.DataFrame) -> None:
    st.subheader("🩺 Facility Explorer")
    st.caption("Browse, filter, and inspect individual facilities with evidence details.")

    if df.empty:
        st.error("No cleaned data found at `data/processed/facilities_clean.csv`.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        state_options = sorted(df["address_state_or_region_clean"].dropna().unique().tolist())
        sel_state = st.selectbox("State", ["All"] + state_options, index=0)
    with c2:
        type_options = sorted(df["facility_type_id"].dropna().unique().tolist())
        sel_type = st.selectbox("Facility type", ["All"] + type_options, index=0)
    with c3:
        text = st.text_input(
            "Name / city search",
            placeholder="Search by facility name or city (e.g., Delhi, Apollo, Fortis)",
        )

    filtered = df.copy()
    if sel_state != "All":
        filtered = filtered[filtered["address_state_or_region_clean"] == sel_state]
    if sel_type != "All":
        filtered = filtered[filtered["facility_type_id"] == sel_type]
    if text.strip():
        q = text.strip().lower()
        filtered = filtered[
            filtered["name"].fillna("").str.lower().str.contains(q)
            | filtered["address_city"].fillna("").str.lower().str.contains(q)
        ]

    view_cols = [
        "name",
        "facility_type_id",
        "address_city",
        "address_state_or_region_clean",
        "pin_code_clean",
        "official_phone_clean",
        "core_fields_present",
    ]
    show_cols = [c for c in view_cols if c in filtered.columns]
    display_df = filtered[show_cols].rename(columns=DISPLAY_COLUMNS)
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "No."
    st.write(f"Showing `{len(filtered)}` facilities")
    st.dataframe(display_df, use_container_width=True, hide_index=False)

    if len(filtered) > 0:
        st.markdown("---")
        st.write("### Facility Detail")
        names = filtered["name"].dropna().tolist()
        selected_name = st.selectbox("Select facility", names, index=0)
        row = filtered[filtered["name"] == selected_name].iloc[0]

        d1, d2 = st.columns([2, 1])
        with d1:
            st.markdown(f"**{selected_name}**")
            st.write(
                f"{_safe_text(row.get('address_city'))}, "
                f"{_safe_text(row.get('address_state_or_region_clean'))}"
            )
            st.write(f"Type: `{_safe_text(row.get('facility_type_id'))}`")
            st.write("Description:")
            st.write(_safe_text(row.get("description"), "No description available."))
        with d2:
            trust_score, completeness, consistency, evidence_strength = _compute_trust_components(row)
            st.metric("Trust Score", f"{trust_score}")
            st.write("Breakdown:")
            st.progress(completeness, text=f"Completeness ({int(completeness * 100)}%)")
            st.progress(consistency, text=f"Consistency ({int(consistency * 100)}%)")
            st.progress(evidence_strength, text=f"Evidence strength ({int(evidence_strength * 100)}%)")

        e1, e2 = st.columns(2)
        with e1:
            _render_structured_list("Specialties", parse_list_cell(row.get("specialties")))
            _render_structured_list("Procedures", parse_list_cell(row.get("procedure")))
        with e2:
            _render_structured_list("Equipment", parse_list_cell(row.get("equipment")))
            _render_structured_list("Capability", parse_list_cell(row.get("capability")))
