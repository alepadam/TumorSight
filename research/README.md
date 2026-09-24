# Research

This folder is for *understanding* the models, not just scoring them.
`notebooks/06_model_comparison.ipynb` answers "which model is more accurate?" —
these notebooks answer "why?", "what did each model actually learn?", and "what
happens if a training choice changes?"

## Notebooks

| Notebook | Question it answers |
|---|---|
| `01_filters_and_feature_maps.ipynb` | What visual patterns has each model's early layers learned to detect? Concretely shows *why* transfer learning helps: a from-scratch CNN's first-layer filters start as random noise, while DenseNet's pretrained filters already detect edges/textures from ImageNet. |
| `02_gradcam_interpretability.ipynb` | For a given MRI, *where* is the model looking when it predicts a class? Grad-CAM heatmaps overlaid on real scans — a sanity check that the model attends to the tumor region, not an unrelated artifact (scanner watermark, image border, etc). |
| `03_training_dynamics_comparison.ipynb` | How did each model's loss/accuracy evolve during training? Overfitting gap (train vs val), and the visible effect of unfreezing DenseNet's layers partway through (phase 1 → phase 2). |
| `04_layer_freezing_ablation.ipynb` | Systematic experiment: train the same transfer model with different numbers of unfrozen layers (0, 10, 30, all) and measure the effect on validation accuracy — turns "unfreeze the last 30 layers" from an arbitrary-sounding default into a measured choice. |

## Relationship to `src/`

Every notebook here imports from `src/interpretability.py`, `src/models.py`, etc. —
same as `notebooks/`. Nothing in `research/` duplicates pipeline logic; it only
*inspects* models and data that the main pipeline already produces. If you find
yourself copy-pasting logic instead of importing it, that's a sign it belongs in
`src/` instead, with a test.

## Prerequisites

These notebooks load trained model checkpoints from `models/saved_models/` (produced
by `notebooks/04` and `notebooks/05`) and the processed dataset from
`data/processed/metadata_split.csv` (produced by `notebooks/01` and `03`). Run the
main pipeline notebooks first.
