# TumorSight

4-class MRI classification (glioma, meningioma, pituitary tumor, no tumor) built with a
patient-level train/val/test split to avoid the data leakage issue present in several
public versions of this dataset.

## Why patient-level split matters

The commonly used Kaggle "Brain Tumor MRI Dataset" splits data at the **slice** level, not
the **patient** level. Since multiple MRI slices come from the same patient's scan volume,
a naive random split lets near-identical slices from one patient appear in both train and
test sets — inflating reported accuracy (published results of 96–99%+ on this dataset
generally do not control for this). This project uses the original
[Jun Cheng (2017) Figshare dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427),
which retains patient IDs, to split by patient instead. See `notebooks/07_leakage_ablation.ipynb`
for a side-by-side comparison of naive vs. patient-level split performance.

## Data sources

| Source | Classes | License / access |
|---|---|---|
| [Jun Cheng (2017), Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) | glioma, meningioma, pituitary | Open, patient IDs included |
| [Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) | no tumor | Open (Kaggle) |

Raw data is **not** committed to this repo (see `.gitignore`) — download instructions are
in `data/README.md`.

## Project structure

```
tumorsight/
├── data/                    # raw + processed data (gitignored, see data/README.md)
├── notebooks/                # exploratory + experiment notebooks, numbered by pipeline stage
├── src/                       # reusable, tested source code
│   ├── data_utils.py          # .mat loading, patient-level split
│   ├── models.py               # model architectures (custom CNN, transfer learning)
│   ├── train.py                 # training loop
│   └── evaluate.py              # metrics, confusion matrix, reports
├── tests/                     # unit tests, incl. leakage regression test
├── models/saved_models/    # trained model checkpoints (gitignored, or tracked via Git LFS)
├── dashboard/                # Streamlit app
├── .github/workflows/       # CI pipeline
├── requirements.txt
├── pyproject.toml            # tool config: black, ruff, mypy, pytest
└── Makefile                   # common commands (setup, lint, test, run)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pre-commit install             # sets up lint/format checks on every commit
```

## Common commands

```bash
make lint       # ruff + black --check
make format     # black (auto-fix)
make test       # pytest, including the patient-leakage regression test
make dashboard  # run the Streamlit app locally
```

## Reproducing the pipeline

1. Download raw data per `data/README.md`
2. Run notebooks `01` → `07` in order (or the equivalent `src/` scripts)
3. Trained models are saved to `models/saved_models/`
4. Launch the dashboard: `make dashboard`

## CI

Every push/PR runs: linting, formatting check, unit tests, and the patient-level split
leakage assertion (`tests/test_data_utils.py`). Model training is **not** run in CI
(compute cost) — training is a manual step; only code correctness and the leakage
safeguard are enforced automatically.

## Results

_To be filled in after Phase 6 (model comparison) — see `notebooks/06_model_comparison.ipynb`._

| Model | Split type | Accuracy | F1 (macro) |
|---|---|---|---|
| Custom CNN | Patient-level | TBD | TBD |
| DenseNet121 (transfer) | Patient-level | TBD | TBD |
| DenseNet121 (transfer) | Naive slice-level (ablation) | TBD | TBD |
