import streamlit as st

from src.services.data_loader import load_data
from src.ui.tabs.agent_chat import render_agent_tab
from src.ui.tabs.crisis_map import render_map_tab
from src.ui.tabs.facility_explorer import render_explorer_tab


def main():
    st.set_page_config(
        page_title="Discovery2Care",
        page_icon="🏥",
        layout="wide",
    )

    df = load_data()

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 0.35rem;
            padding-bottom: 1.2rem;
            max-width: 1200px;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        .hero-card {
            background: linear-gradient(135deg, #0b3d91 0%, #1d4ed8 60%, #2563eb 100%);
            color: white;
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 10px;
            box-shadow: 0 8px 24px rgba(29, 78, 216, 0.22);
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 2px;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            opacity: 0.95;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem;
            margin-top: 0.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1rem;
            font-weight: 600;
            padding: 0.8rem 1.15rem;
            border-radius: 10px 10px 0 0;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        .stMetric {
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 12px;
            padding: 0.45rem 0.8rem 0.65rem 0.8rem;
            background: rgba(248, 250, 252, 0.75);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Discovery2Care - India Healthcare Intelligence</div>
            <div class="hero-subtitle">
                Agentic analytics for faster Discovery-to-Care decisions
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["💬 Agent Chat", "🗺️ Crisis Map", "🩺 Facility Explorer"]
    )

    with tab1:
        render_agent_tab(df)
    with tab2:
        render_map_tab(df)
    with tab3:
        render_explorer_tab(df)


if __name__ == "__main__":
    main()
