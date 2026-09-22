"""
Streamlit dashboard entry point. Uses a top nav bar (not sidebar page list) and a
muted color scheme, matching the pattern from the diabetes risk project. Sidebar
is kept for branding only.

Run with: streamlit run dashboard/app.py  (or `make dashboard`)
"""

import streamlit as st

st.set_page_config(
    page_title="TumorSight",
    page_icon="🧠",
    layout="wide",
)

# --- Sidebar: branding only ---
with st.sidebar:
    st.markdown("### 🧠 TumorSight")
    st.caption("4-class MRI classification — glioma, meningioma, pituitary, no tumor")

# --- Top nav bar ---
PAGES = ["Overview", "Tumor Classifier", "Model Performance", "Tumor Info"]
selected_page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed")
st.divider()

if selected_page == "Overview":
    from pages import overview

    overview.render()
elif selected_page == "Tumor Classifier":
    from pages import tumor_classifier

    tumor_classifier.render()
elif selected_page == "Model Performance":
    from pages import model_performance

    model_performance.render()
elif selected_page == "Tumor Info":
    from pages import tumor_info

    tumor_info.render()
