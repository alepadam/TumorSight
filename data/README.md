# Data

Raw data is not committed to this repo — download it yourself and place it as described below.
This keeps the repo small and respects each dataset's own distribution terms.

## 1. Figshare (Cheng, 2017) — tumor classes

- Source: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
- Download all 4 zip parts, extract into `data/raw/figshare_mat/`
- Each `.mat` file contains a struct with:
  - `cjdata.label` — 1=meningioma, 2=glioma, 3=pituitary
  - `cjdata.PID` — patient ID (critical for the patient-level split)
  - `cjdata.image` — pixel data
  - `cjdata.tumorMask` — binary segmentation mask (not used in this project's
    classification scope, kept for a possible future segmentation extension)

## 2. Br35H — no-tumor class

- Source: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection
- Place "no tumor" images into `data/raw/br35h_no_tumor/`
- These images do not have a natural patient ID; `data_utils.py` assigns each a unique
  synthetic ID so they still work with `GroupShuffleSplit` without being able to leak
  (since each one is its own group).

## Processed data

Running `notebooks/01_data_loading_and_merge.ipynb` (or `src/data_utils.py` directly)
produces `data/processed/metadata.csv` — the single source of truth used throughout the
rest of the pipeline, with columns: `image_path, label, patient_id, source_dataset, split`.

## A note on the "no tumor" merge

Br35H images may differ in acquisition style/quality from the Figshare tumor slices.
Check for this explicitly in EDA (image size, contrast, intensity distribution) — if the
model can tell classes apart purely by *which dataset an image came from* rather than by
tumor presence, that's a second, subtler leakage problem worth catching before training.
