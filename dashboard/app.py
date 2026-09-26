"""Streamlit dashboard para auto-eval-platform."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="auto-eval-platform", layout="wide")

st.title("🤖 auto-eval-platform Dashboard")

# Sidebar
project_id = st.sidebar.selectbox("Project", ["demo-project-a", "demo-project-b", "demo-project-c"])
run_id = st.sidebar.selectbox("Run", ["latest", "run_abc123", "run_def456"])

# Mock data for demo
import numpy as np
metrics_data = {
    "Metric": ["Classification Accuracy", "Extraction F1", "Derivation Rate", "Latency P95 (ms)", "Cost per 1k tokens ($)"],
    "Pessimistic": [0.92, 0.89, 0.12, 1800, 0.045],
    "Base": [0.95, 0.93, 0.08, 1200, 0.032],
    "Optimistic": [0.97, 0.96, 0.05, 800, 0.025],
    "Gate": ["✅ PASS", "✅ PASS", "✅ PASS", "✅ PASS", "✅ PASS"]
}
df = pd.DataFrame(metrics_data)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📊 3-Scenario Metrics")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart
    fig = px.bar(df.melt(id_vars=["Metric", "Gate"], value_vars=["Pessimistic", "Base", "Optimistic"]),
                 x="Metric", y="value", color="variable", barmode="group",
                 title="Bootstrap Confidence Intervals")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Gates")
    for _, row in df.iterrows():
        st.metric(row["Metric"], row["Base"], delta=f"Gate: {row['Gate']}")

    st.subheader("🤖 Agents Status")
    agents = [
        ("Golden Set Agent", "🟢 Healthy", "v2.1 • kappa=0.84"),
        ("Red Team Agent", "🟢 Healthy", "Last run: 02:00 UTC • 0 critical"),
        ("Gate Synthesis", "🟢 Healthy", "Updated 1h ago"),
        ("Drift Agent", "🟡 Warning", "PSI=0.18 (threshold 0.20)"),
        ("Compliance Agent", "🟢 Healthy", "Ley 25.326 • GDPR • EU AI Act"),
    ]
    for name, status, detail in agents:
        st.write(f"**{name}**: {status}  \n<small>{detail}</small>", unsafe_allow_html=True)

st.subheader("📈 Drift Monitoring (PSI/KL)")
fig = go.Figure()
fig.add_trace(go.Scatter(y=np.cumsum(np.random.randn(30)*0.01), name="PSI", line=dict(color="#00d4aa")))
fig.add_trace(go.Scatter(y=[0.2]*30, name="Threshold", line=dict(color="#ef4444", dash="dash")))
st.plotly_chart(fig, use_container_width=True)
