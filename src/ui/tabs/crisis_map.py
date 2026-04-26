import pandas as pd
import plotly.express as px
import streamlit as st

from src.services.parsers import parse_list_cell


NEED_KEYWORDS = {
    "Emergency Trauma": [
        "emergency",
        "trauma",
        "critical care",
        "appendectomy",
        "general surgery",
        "anesthesia",
    ],
    "Oncology": [
        "oncology",
        "cancer",
        "tumour",
        "tumor",
        "chemotherapy",
        "radiation",
    ],
    "Dialysis": [
        "dialysis",
        "dialys",
        "nephrology",
        "renal",
        "kidney",
        "hemodialysis",
        "hemodialysis",
        "peritoneal",
    ],
    "ICU": [
        "icu",
        "intensive care",
        "critical care",
        "ventilator",
        "emergency",
    ],
    "Neonatal": [
        "neonatal",
        "nicu",
        "newborn",
        "pediatric",
        "paediatric",
        "obstetrics",
    ],
}


def _build_search_blob(row: pd.Series) -> str:
    parts = [
        str(row.get("description", "")),
        " ".join(parse_list_cell(row.get("specialties"))),
        " ".join(parse_list_cell(row.get("procedure"))),
        " ".join(parse_list_cell(row.get("equipment"))),
        " ".join(parse_list_cell(row.get("capability"))),
    ]
    return " ".join(parts).lower()


def _match_need(row: pd.Series, need: str) -> bool:
    blob = _build_search_blob(row)
    return any(keyword in blob for keyword in NEED_KEYWORDS.get(need, []))

def _pretty_facility_type(value: str) -> str:
    txt = str(value).strip().replace("_", " ")
    return txt.title() if txt else "Unknown"

def _need_match_label(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def render_map_tab(df: pd.DataFrame) -> None:
    st.subheader("🗺️ Crisis Map")
    st.caption("Analyze healthcare service availability by need and area.")

    if df.empty:
        st.error("No cleaned data found at `data/processed/facilities_clean.csv`.")
        return

    left, right = st.columns([1, 3])
    with left:
        need = st.selectbox(
            "Medical need",
            ["Emergency Trauma", "Oncology", "Dialysis", "ICU", "Neonatal"],
        )
        state_options = sorted(df["address_state_or_region_clean"].dropna().unique().tolist())
        selected_state = st.selectbox("State/Region", ["All"] + state_options, index=0)
        severity_mode = st.radio("View mode", ["Coverage map", "High-risk focus"])
        st.write("**Legend**")
        st.markdown("- 🔵 Matched facilities")
        st.markdown("- ⚪ Other facilities in selected state")
        st.markdown("- Use State/Region filter to focus area")

    with right:
        plot_df = df.copy()
        if selected_state != "All":
            plot_df = plot_df[plot_df["address_state_or_region_clean"] == selected_state]

        if plot_df.empty:
            st.warning("No facilities found for selected area filters.")
            return

        matched_mask = plot_df.apply(lambda row: _match_need(row, need), axis=1)
        plot_df = plot_df.assign(need_match=matched_mask)
        matched_in_area = int(matched_mask.sum())
        total_in_area_before_mode = int(len(plot_df))

        if severity_mode == "High-risk focus":
            plot_df = plot_df[plot_df["need_match"]]
            if plot_df.empty:
                st.warning(
                    "No facilities matched this medical need in the selected area. "
                    "Try Coverage map mode or broaden the area filter."
                )
                return

        for col in ["latitude", "longitude"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=["latitude", "longitude"]).head(2500)
        if "facility_type_id" not in plot_df.columns:
            plot_df["facility_type_id"] = "unknown"
        plot_df["facility_type_label"] = plot_df["facility_type_id"].map(_pretty_facility_type)
        plot_df["need_match_label"] = plot_df["need_match"].map(_need_match_label)

        title_area = selected_state if selected_state != "All" else "India"

        # Auto-zoom behavior: when state filter is applied, zoom map to filtered bounds.
        lat_range = [6, 38]
        lon_range = [68, 98]
        if selected_state != "All":
            lat_min, lat_max = plot_df["latitude"].min(), plot_df["latitude"].max()
            lon_min, lon_max = plot_df["longitude"].min(), plot_df["longitude"].max()
            lat_pad = max((lat_max - lat_min) * 0.38, 1.0)
            lon_pad = max((lon_max - lon_min) * 0.38, 1.2)
            lat_range = [max(6, lat_min - lat_pad), min(38, lat_max + lat_pad)]
            lon_range = [max(68, lon_min - lon_pad), min(98, lon_max + lon_pad)]

        fig = px.scatter_geo(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="facility_type_label",
            hover_name="name",
            hover_data={
                "address_city": True,
                "address_state_or_region_clean": True,
                "facility_type_label": True,
                "need_match_label": True,
            },
            labels={"facility_type_label": "Facility Type", "need_match_label": "Need Match"},
            title=f"{need} coverage view - {title_area}",
            projection="natural earth",
            height=650,
        )
        fig.update_geos(
            showcountries=True,
            showsubunits=True,
            subunitcolor="#cbd5e1",
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#eaf2ff",
            countrycolor="#94a3b8",
            lataxis_range=lat_range,
            lonaxis_range=lon_range,
        )
        fig.update_traces(marker={"size": 6, "opacity": 0.85})
        fig.update_layout(legend_title_text="Facility Type")
        st.plotly_chart(fig, use_container_width=True)

        total_area = total_in_area_before_mode
        matched_area = matched_in_area
        st.caption(
            f"Matched facilities in current view: {matched_area} / {total_area}"
        )

    st.markdown("### Coverage Summary by State/Region")
    summary_df = df.copy()
    summary_df["need_match"] = summary_df.apply(lambda row: _match_need(row, need), axis=1)
    grouped = (
        summary_df.groupby("address_state_or_region_clean", dropna=True)["need_match"]
        .agg(total="count", matched="sum")
        .reset_index()
        .rename(columns={"address_state_or_region_clean": "State/Region"})
    )
    grouped["Coverage %"] = (grouped["matched"] / grouped["total"] * 100).round(1)
    grouped["Risk Level"] = grouped["Coverage %"].apply(
        lambda v: "High risk" if v < 20 else ("Medium risk" if v < 40 else "Better served")
    )
    grouped = grouped.sort_values(by=["matched", "Coverage %", "total"], ascending=[False, False, False])
    grouped_top = grouped.head(20).reset_index(drop=True)
    grouped_top.index = range(1, len(grouped_top) + 1)
    grouped_top.index.name = "No."
    st.dataframe(grouped_top, use_container_width=True, hide_index=False)
