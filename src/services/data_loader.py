import pandas as pd
import streamlit as st

from src.config.constants import DATA_PATH


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()
