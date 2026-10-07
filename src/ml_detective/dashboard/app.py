"""

Streamlit dashboard -- calls ml_detective modules DIRECTLY (same
process, no HTTP/API layer). Run with: streamlit run src/ml_detective/dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ml_detective.db.repository import list_investigations, save_investigation
from ml_detective.ingestion.task_detector import guess_target_column
from ml_detective.orchestrator import run_full_investigation
from ml_detective.reporting.report_builder import generate_report
from ml_detective.config.settings import settings

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
st.set_page_config(page_title="ML Detective", page_icon="🔍", layout="wide")
st.title("🔍 ML Detective")
st.caption("Don't train the model until the detective finishes the investigation.")

if "results" not in st.session_state:
    st.session_state.results = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.session_state.dataframe = dataframe

    st.write(f"**{dataframe.shape[0]} rows × {dataframe.shape[1]} columns**")
    st.dataframe(dataframe.head(5), use_container_width=True)

    guessed_target = guess_target_column(dataframe)
    target_column = st.selectbox(
        "Target column (what are you predicting?)",
        options=[None] + list(dataframe.columns),
        index=(list(dataframe.columns).index(guessed_target) + 1) if guessed_target else 0,
    )

    use_llm = st.checkbox("Use local LLM (Ollama) for report narration", value=True)

    if st.button("🔎 Run Investigation", type="primary"):
        with st.spinner("Investigating dataset..."):
            results = run_full_investigation(dataframe, target_column=target_column)
            results["report"] = generate_report(
                results["profile"], results["health_report"], results["detective_findings"], use_llm=use_llm
            )
        st.session_state.results = results
        save_investigation(
            dataset_name=uploaded_file.name,
            n_rows=dataframe.shape[0],
            n_cols=dataframe.shape[1],
            target_column=results["target_column"],
            task_type=results["task_type"],
            health_report=results["health_report"],
            detective_findings=results["detective_findings"],
            report=results["report"],
        )

if st.session_state.results:
    results = st.session_state.results
    health = results["health_report"]
    detective_findings = results["detective_findings"]

    tab_overview, tab_findings, tab_report, tab_explorer,tab_history = tab_overview, tab_findings, tab_report, tab_explorer, tab_history = st.tabs(
        ["📊 Overview", "🕵️ Findings", "📄 Report", "🔬 Feature Explorer", "🕐 History"]
    )

    with tab_overview:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Health Score", f"{health.overall_score}/100", health.grade)
            st.metric("Target Column", results["target_column"] or "Not detected")
            st.metric("Task Type", results["task_type"])
            st.metric("Total Findings", len(detective_findings))

        with col2:
            dims = list(health.dimension_scores.keys())
            scores = [health.dimension_scores[d].score for d in dims]
            fig = go.Figure(go.Bar(x=dims, y=scores, marker_color="indianred"))
            fig.update_layout(title="Health Score by Dimension", yaxis_range=[0, 100], height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab_findings:
        severity_filter = st.multiselect(
            "Filter by severity", options=["critical", "high", "medium", "low"],
            default=["critical", "high", "medium", "low"],
        )
        search_term = st.text_input("Search findings")

        filtered = [
            f for f in detective_findings
            if f.severity.value in severity_filter
            and (search_term.lower() in f.finding.lower() if search_term else True)
        ]

        severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        for f in sorted(filtered, key=lambda x: ["critical", "high", "medium", "low"].index(x.severity.value)):
            with st.expander(f"{severity_icon[f.severity.value]} {f.finding} (confidence: {f.confidence})"):
                st.write(f"**Reasoning:** {f.reasoning}")
                st.write(f"**Business Impact:** {f.business_impact}")
                st.write(f"**Model Impact:** {f.model_impact}")
                st.write(f"**Recommended Fix:** {f.recommended_fix}")
                st.json(f.evidence)

        if not filtered:
            st.info("No findings match the current filter.")

    with tab_report:
        report = results["report"]
        st.subheader("Executive Summary")
        st.write(report["executive_summary"])
        st.subheader("Critical Findings")
        st.write(report["critical_findings"])
        st.subheader("Medium Findings")
        st.write(report["medium_findings"])
        st.subheader("Low Findings")
        st.write(report["low_findings"])
        st.subheader("Next Steps")
        st.write(report["next_steps"])
        st.caption(f"Generated with LLM: {report['generated_with_llm']} (provider configured: {settings.llm_provider or 'ollama'})")

        report_markdown = "\n\n".join(
            f"## {k.replace('_', ' ').title()}\n{v}" for k, v in report.items() if k != "generated_with_llm"
        )
        st.download_button("⬇️ Download Report (Markdown)", report_markdown, file_name="ml_detective_report.md")

    with tab_explorer:
        profile = results["profile"]
        rows = [p.to_dict() for p in profile.column_profiles.values()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab_history:
        past = list_investigations()
        if not past:
            st.info("No past investigations yet.")
        for inv in past:
            st.write(f"**{inv.dataset_name}** — {inv.health_score}/100 ({inv.grade}) — {inv.created_at}")