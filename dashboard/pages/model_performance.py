import streamlit as st


def render():
    st.header("Model Performance")
    st.write(
        "Comparison of the custom CNN baseline vs. DenseNet121 transfer learning, "
        "both trained on a patient-level split."
    )
    # TODO: load metrics from notebooks/06_model_comparison.ipynb output, display:
    # - accuracy / precision / recall / F1 table per model
    # - confusion matrix heatmap per model
    # - the leakage ablation: naive split vs patient-level split accuracy, side by side
    st.info("Populate this page after Phase 6 (model comparison) is complete.")
