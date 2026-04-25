import pandas as pd
import streamlit as st

from src.services.parsers import parse_list_cell


def render_explorer_tab(df: pd.DataFrame) -> None:
    st.subheader("🩺 Facility Explorer")
    st.caption("Browse, filter, and inspect individual facilities with evidence details.")

    if df.empty:
        st.error("No cleaned data found at `data/processed/facilities_clean.csv`.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        state_options = sorted(df["address_state_or_region_clean"].dropna().unique().tolist())
        sel_state = st.selectbox("State", ["All"] + state_options)
    with c2:
        type_options = sorted(df["facility_type_id"].dropna().unique().tolist())
        sel_type = st.selectbox("Facility type", ["All"] + type_options)
    with c3:
        text = st.text_input("Name / city search")

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
    st.write(f"Showing `{len(filtered)}` facilities")
    st.dataframe(filtered[show_cols].head(300), use_container_width=True, hide_index=True)

    if len(filtered) > 0:
        st.markdown("---")
        st.write("### Facility Detail")
        names = filtered["name"].dropna().tolist()
        selected_name = st.selectbox("Select facility", names)
        row = filtered[filtered["name"] == selected_name].iloc[0]

        d1, d2 = st.columns([2, 1])
        with d1:
            st.markdown(f"**{selected_name}**")
            st.write(
                f"{row.get('address_city', 'N/A')}, "
                f"{row.get('address_state_or_region_clean', 'N/A')}"
            )
            st.write(f"Type: `{row.get('facility_type_id', 'N/A')}`")
            st.write("Description:")
            st.write(row.get("description", "No description available."))
        with d2:
            st.metric("Trust Score", "74")
            st.write("Breakdown:")
            st.progress(0.82, text="Completeness")
            st.progress(0.69, text="Consistency")
            st.progress(0.71, text="Evidence strength")

        e1, e2 = st.columns(2)
        with e1:
            st.write("Specialties")
            st.json(parse_list_cell(row.get("specialties")))
            st.write("Procedures")
            st.json(parse_list_cell(row.get("procedure")))
        with e2:
            st.write("Equipment")
            st.json(parse_list_cell(row.get("equipment")))
            st.write("Capability")
            st.json(parse_list_cell(row.get("capability")))
