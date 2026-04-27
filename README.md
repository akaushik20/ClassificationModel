# Binary Classification Pipeline

An end-to-end binary classification pipeline built with **Python 3.10**. The project uses a fully config-driven architecture — every path, threshold, and model parameter is defined in `config/config.yaml` with no hardcoded values in the source code.

---

## Project Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `helper/data_generation.py` | Generates a synthetic 10k-row dataset with mixed types, missingness, and a 70/30 class imbalance |
| 2 | `helper/eda.py` | Produces an automated EDA HTML report using Sweetviz |
| 3 | `helper/data_prep.py` | Cleans, encodes, scales, and splits the data into train/test sets |
| 4 | `helper/train.py` | Trains XGBoost, selects top-N features by importance, and saves the model |
| 5 | `helper/evaluate.py` | Generates evaluation plots and a metrics summary CSV |

`main.py` orchestrates all five stages in sequence.

---

## Folder Structure

```
ClassificationModel/
│
├── main.py                        # Full pipeline orchestrator
├── README.md
├── requirements.txt
│
├── config/
│   └── config.yaml                # All paths, params & thresholds (single source of truth)
│
├── helper/
│   ├── __init__.py
│   ├── data_generation.py         # Step 1 — synthetic data
│   ├── eda.py                     # Step 2 — Sweetviz EDA report
│   ├── data_prep.py               # Step 3 — preprocessing pipeline
│   ├── train.py                   # Step 4 — XGBoost training + feature selection
│   └── evaluate.py                # Step 5 — metrics, plots, summary CSV
│
├── data/                          # Raw + processed CSVs (git-ignored except .gitkeep)
├── model/                         # Saved .pkl artifacts (git-ignored except .gitkeep)
└── output/                        # EDA report, plots, metrics CSV
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ClassificationModel.git
cd ClassificationModel
```

### 2. Create and activate a Python 3.10 virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the full pipeline end-to-end with a single command:

```bash
python main.py
```

Each stage logs its progress to the console. On completion you will find:

| Artifact | Location |
|----------|----------|
| Raw dataset | `data/raw.csv` |
| Processed train set | `data/train.csv` |
| Processed test set | `data/test.csv` |
| EDA report | `output/eda_report.html` |
| Trained model | `model/model.pkl` |
| Selected feature list | `model/features.pkl` |
| Confusion matrix | `output/confusion_matrix.png` |
| ROC-AUC curve | `output/roc_curve.png` |
| Precision-Recall curve | `output/pr_curve.png` |
| Feature importance chart | `output/feature_importance.png` |
| Metrics summary | `output/metrics_summary.csv` |

---

## Configuration

All tunable parameters live in [`config/config.yaml`](config/config.yaml). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `random_seed` | `42` | Global reproducibility seed |
| `preprocessing.missing_drop_threshold` | `0.90` | Drop columns with >90% missing |
| `preprocessing.test_size` | `0.20` | Train/test split ratio |
| `feature_selection.top_n_features` | `15` | Features retained after importance ranking |
| `model.n_estimators` | `300` | Number of XGBoost trees |
| `model.learning_rate` | `0.05` | XGBoost learning rate |
| `evaluation.classification_threshold` | `0.50` | Probability threshold for class prediction |

---

## Design Decisions

- **Sweetviz over ydata-profiling** — 5–10× faster on 10k rows, no heavy optional dependencies, and produces a clean target-correlation-focused report ideal for classification tasks.
- **StandardScaler** — synthetic numeric features are approximately normal with no extreme skew; StandardScaler (zero mean, unit variance) is the appropriate choice over MinMaxScaler.
- **XGBoost feature importance (gain-based)** — model-native, fast, handles mixed types after encoding, and is consistent with the final estimator; no additional estimator required for selection.
- **`scale_pos_weight` for class imbalance** — computed dynamically as `n_negative / n_positive`; avoids synthetic over-sampling artefacts (SMOTE) on already-synthetic data.
