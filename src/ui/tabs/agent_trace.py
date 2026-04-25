import pandas as pd
import streamlit as st


def render_trace_tab() -> None:
    st.subheader("⚙️ Agent Trace")
    st.caption("Structured step-by-step execution trace for transparency.")

    st.write("### Last Query")
    st.code("Find facilities in rural Bihar that can perform emergency appendectomy and have high trust score.")

    st.write("### Execution Timeline")
    trace_rows = [
        {"Step": "1. Query normalize", "Status": "Done", "Duration (ms)": 42, "Output": "Parsed location + procedure intent"},
        {"Step": "2. Retrieve candidates", "Status": "Done", "Duration (ms)": 190, "Output": "287 facilities matched initial keywords"},
        {"Step": "3. Capability scoring", "Status": "Done", "Duration (ms)": 156, "Output": "Reduced to top 25 relevance"},
        {"Step": "4. Trust scoring", "Status": "Done", "Duration (ms)": 91, "Output": "Assigned trust buckets High/Med/Low"},
        {"Step": "5. Final synthesis", "Status": "Done", "Duration (ms)": 65, "Output": "Returned top 3 with evidence"},
    ]
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)

    st.write("### Tool Calls")
    st.code(
        "load_cleaned_data() -> success\n"
        "keyword_candidate_filter(query='appendectomy bihar') -> 287\n"
        "rank_candidates() -> 25\n"
        "assemble_response(top_k=3) -> success"
    )

    st.write("### Evidence Links")
    ev = pd.DataFrame(
        [
            {"Facility": "72 BPM Healthcare, Multi Specialty Hospital", "Field": "procedure", "Snippet": "Emergency appendectomy..."},
            {"Facility": "7 Star Healthcare (Hospital)", "Field": "capability", "Snippet": "Always open..."},
            {"Facility": "3D Plus Maxillofacial Imaging", "Field": "equipment", "Snippet": "CBCT scanner..."},
        ]
    )
    st.dataframe(ev, use_container_width=True, hide_index=True)
    st.info("MLflow trace integration will be connected in the next iteration.")
