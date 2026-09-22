import streamlit as st

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]


def render():
    st.header("Tumor Classifier")
    uploaded_file = st.file_uploader("Upload an MRI scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI", width=300)
        # TODO:
        # 1. Load trained model from models/saved_models/
        # 2. Preprocess uploaded image to match training pipeline (resize, normalize)
        # 3. Run prediction, get per-class confidence scores
        # 4. Display predicted class + a confidence bar chart (st.bar_chart)
        st.info("Model inference not yet wired up — connect models/saved_models/ here.")
