import streamlit as st


def render():
    st.header("Overview")
    st.write(
        "This project classifies brain MRI scans into four categories: glioma, "
        "meningioma, pituitary tumor, or no tumor — trained with a patient-level "
        "train/test split to avoid data leakage present in common public versions "
        "of this dataset."
    )
    # TODO: dataset summary stats, class distribution chart, sample MRI grid
