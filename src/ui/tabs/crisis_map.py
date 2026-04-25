import pandas as pd
import plotly.express as px
import streamlit as st


def render_map_tab(df: pd.DataFrame) -> None:
    st.subheader("🗺️ Crisis Map")
    st.caption("Visualize service coverage gaps and underserved zones across India.")

    left, right = st.columns([1, 3])
    with left:
        need = st.selectbox(
            "Medical need",
            ["Emergency Trauma", "Oncology", "Dialysis", "ICU", "Neonatal"],
        )
        severity_mode = st.radio("Desert metric", ["High risk zones", "Coverage density"])
        st.write("**Legend**")
        st.markdown("- 🔴 High risk")
        st.markdown("- 🟡 Moderate risk")
        st.markdown("- 🟢 Better served")

    with right:
        if df.empty:
            st.error("No cleaned data found at `data/processed/facilities_clean.csv`.")
            return

        plot_df = df.copy()
        for col in ["latitude", "longitude"]:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
        plot_df = plot_df.dropna(subset=["latitude", "longitude"]).head(2000)
        if "facility_type_id" not in plot_df.columns:
            plot_df["facility_type_id"] = "unknown"

        fig = px.scatter_geo(
            plot_df,
            lat="latitude",
            lon="longitude",
            color="facility_type_id",
            hover_name="name",
            hover_data={"address_city": True, "address_state_or_region_clean": True},
            title=f"India Facility Distribution ({need} - {severity_mode})",
            projection="natural earth",
            height=650,
        )
        fig.update_geos(
            showcountries=True,
            countrycolor="LightGray",
            lataxis_range=[6, 38],
            lonaxis_range=[68, 98],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Underserved Zone Highlights")
    zone_df = pd.DataFrame(
        [
            {"Zone": "North Bihar Cluster", "Need": need, "Severity": "High", "Facilities matched": 12},
            {"Zone": "Western Rajasthan Belt", "Need": need, "Severity": "High", "Facilities matched": 9},
            {"Zone": "Central MP Rural Block", "Need": need, "Severity": "Medium", "Facilities matched": 21},
        ]
    )
    st.dataframe(zone_df, use_container_width=True, hide_index=True)
