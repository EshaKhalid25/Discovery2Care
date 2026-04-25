import pandas as pd
import streamlit as st

from src.services.mock_agent import build_mock_agent_response


def render_agent_tab(df: pd.DataFrame) -> None:
    st.subheader("💬 Agent Chat")
    st.caption("Search facilities with multi-condition healthcare queries.")

    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input(
            "Ask your healthcare query",
            placeholder="Find facilities in rural Bihar that can perform emergency appendectomy and have strong evidence.",
        )
    with c2:
        ask_clicked = st.button("Run Query", type="primary", use_container_width=True)

    st.write("**Try examples:**")
    ex_cols = st.columns(3)
    example_queries = [
        "Emergency appendectomy near rural Bihar with 24/7 signals",
        "Oncology deserts in northern states",
        "Dialysis-ready facilities with higher trust score",
    ]
    for idx, col in enumerate(ex_cols):
        col.button(example_queries[idx], use_container_width=True, disabled=True)

    st.markdown("---")
    st.write("### Agent Response")

    if ask_clicked or query:
        active_query = query or "Emergency appendectomy near rural Bihar with 24/7 signals"
        st.info(f"Query processed: `{active_query}`")
        responses = build_mock_agent_response(active_query)
        for item in responses:
            with st.container(border=True):
                head1, head2, head3 = st.columns([3, 1, 1])
                head1.markdown(f"**{item['facility']}**  \n{item['location']}")
                head2.metric("Match", f"{item['match_score']}%")
                head3.metric("Trust", item["trust_label"])
                st.write(item["why"])
                with st.expander("View evidence snippet"):
                    st.code(item["evidence"])
    else:
        st.warning("Enter a query and click Run Query to see recommendations.")
