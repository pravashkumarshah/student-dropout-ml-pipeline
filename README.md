# End-to-End Machine Learning Pipeline for Predicting Student Dropout and Academic Success

University machine learning project to build a clean, reproducible, end-to-end pipeline that predicts student dropout and academic success.

> Week 1 status: initial scaffold only. No dataset, no training, no results yet.

## Goals
- Explore factors associated with student dropout and academic success.
- Build a preprocessing + feature engineering pipeline in `src/`.
- Train and compare baseline classifiers in a reproducible way.
- Evaluate with proper validation (no leakage, no fake metrics).
- Document findings in `reports/` in Week 6.

## Project Structure
```
prgt/
├── data/           # Dataset (not committed). Raw/processed CSVs go here. Currently empty.
├── notebooks/      # Exploratory analysis and experiments (Week 2+).
├── src/            # Reusable Python code (preprocessing, training, evaluation).
├── models/         # Saved models (.pkl/.joblib, git-ignored). Currently empty.
├── reports/        # Figures and final report (Week 6).
├── tests/          # pytest tests for src utilities.
├── README.md
├── requirements.txt
└── .gitignore
```

Empty folders contain `.gitkeep` so they stay GitHub-ready. `src/__init__.py` and `tests/__init__.py` make packages importable.

## Setup (Python)

```powershell
# From project root: c:\Users\Pravash kumar shah\OneDrive\Desktop\prgt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires Python 3.10+ (recommended). Do not commit `.venv/`.

## Six-Week Development Plan
- **Week 1 - Project planning:** Define problem, success metrics, repo scaffold. See `reports/week1_project_plan.md` (Week 1 planning package).
- **Week 2 - Data preprocessing and feature engineering:** Load data, clean missing values, encode categoricals, scale numerics, document pipeline in `notebooks/` + `src/`.
- **Week 3 - Model implementation:** Baseline models (e.g. logistic regression, tree-based) with train/validation split in `src/`.
- **Week 4 - Evaluation and validation:** Cross-validation, confusion matrix, precision/recall/F1, ROC-AUC. No test leakage.
- **Week 5 - Model optimization:** Hyperparameter tuning, feature selection, error analysis.
- **Week 6 - Final report and comprehensive analysis:** Write up methodology, results, limitations, future work in `reports/`.

## Data Notes
- Dataset is **not included** in this repo and has not been downloaded yet.
- When added, place raw files under `data/` (git-ignored: `data/*.csv`, `data/*.xlsx`, `data/*.parquet`).
- Never commit personal / identifiable student data.

## Models Notes
- Saved artifacts (`*.pkl`, `*.joblib`, `*.onnx`, etc.) are git-ignored and not committed.
- Keep training reproducible: fix `random_state`, log params, save only from `src/` scripts.

## Testing
```powershell
pytest
```

## Contributing
- Keep the repository clean and GitHub-ready.
- Do not commit data, models, secrets (`.env`), or IDE cache.
- See `.gitignore` (preserved from initial commit + minimal ML additions).
