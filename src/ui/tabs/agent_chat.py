import pandas as pd
import streamlit as st

from src.services.agent_chat_engine import run_agent_query
from src.services.databricks_client import databricks_status
from src.services.mlflow_tracker import log_agent_query_run


def render_agent_tab(df: pd.DataFrame) -> None:
    st.subheader("💬 Agent Chat")

    if "agent_chat_result" not in st.session_state:
        st.session_state.agent_chat_result = None

    st.text_area(
        "Message",
        height=88,
        key="agent_chat_message",
        placeholder="Describe what care you are looking for, location, and urgency…",
        label_visibility="collapsed",
    )
    send = st.button("Send", type="primary", use_container_width=True)

    if send:
        q = str(st.session_state.get("agent_chat_message", "")).strip()
        if not q:
            st.warning("Please enter a question.")
        elif df.empty:
            st.error("No cleaned dataset loaded. Please run data cleaning first.")
        else:
            with st.spinner("Sending request… searching facilities and generating AI summary."):
                out = run_agent_query(df, q)
                rid = log_agent_query_run(q, out)
                if rid:
                    out["mlflow_run_id"] = rid
            st.session_state.agent_chat_result = out
            st.rerun()

    st.markdown("---")
    st.markdown("### Response")
    result = st.session_state.agent_chat_result
    if result is None:
        st.caption("Results appear here after you send a question.")
    else:
        run_id = str(result.get("mlflow_run_id", "") or "").strip()

        i1, i2, i3 = st.columns(3)
        i1.metric("Need Type", str(result["need"]).title())
        i2.metric("State Filters", len(result.get("states", [])))
        i3.metric("Results Returned", len(result.get("results", [])))
        st.caption(f"Engine: {result.get('engine', 'Local')}")
        if run_id:
            with st.expander("Run metadata", expanded=False):
                st.caption(f"MLflow Run ID: `{run_id}`")

        llm_err = str(result.get("llm_error", "") or "").strip()
        groq_on = databricks_status().get("groq_configured", False)
        if groq_on and llm_err:
            st.warning(f"Summary could not be generated: `{llm_err}`")

        llm_summary = result.get("llm_summary", "")
        if llm_summary:
            st.success(llm_summary)

        with st.expander("Query Understanding", expanded=False):
            st.write(f"**Detected need:** {str(result['need']).title()}")
            st.write(f"**States:** {', '.join(result.get('states', [])) or 'None detected'}")
            st.write(f"**Cities:** {', '.join(result.get('cities', [])) or 'None detected'}")
            st.write(f"**Keywords used:** {', '.join(result.get('keywords', [])) or 'General search'}")
            fallback_mode = result.get("fallback_mode", "strict")
            if fallback_mode == "databricks_primary":
                st.write("**Search mode:** Databricks primary retrieval")
            elif fallback_mode == "strict":
                st.write("**Search mode:** Strict (primary match)")
            elif fallback_mode == "relaxed_area":
                st.write("**Search mode:** Relaxed area fallback (broader geography)")
            elif "relaxed_need" in fallback_mode:
                st.write("**Search mode:** Relaxed need fallback (broader relevance)")

        responses = result.get("results", [])
        if not responses:
            st.warning("No matching facilities found. Try a broader query or different wording.")
        else:
            for item in responses:
                with st.container(border=True):
                    head1, head2, head3 = st.columns([3, 1, 1])
                    head1.markdown(f"**{item['facility']}**  \n{item['location']}")
                    head2.metric("Match", f"{item['match_score']}%")
                    head3.metric("Trust", item["trust_label"])
                    st.write(item["why"])
                    with st.expander("View evidence"):
                        for ev in item["evidence"]:
                            st.markdown(f"- {ev}")
