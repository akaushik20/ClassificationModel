# Project Brief — Take-Home Data Science Interview

> Original prompt used to define the project scope and phased implementation plan.

---

## Project Requirements

Build a clean, well-structured Python repository that fulfills the following requirements.

---

## Project Structure

```
project/
│
├── main.py                  # Orchestrates the full pipeline end-to-end
├── README.md                # Project overview, setup instructions, how to run
│
├── config/
│   └── config.yaml          # Centralized config (paths, hyperparameters, thresholds, random seed, etc.)
│
├── helper/
│   ├── data_generation.py   # Generates synthetic dataset
│   ├── eda.py               # Automated EDA using Sweetviz
│   ├── data_prep.py         # Preprocessing pipeline
│   ├── train.py             # Model training, saves .pkl to /model
│   └── evaluate.py          # Evaluation metrics + saves plots to /output
│
├── model/                   # Saved model artifacts (.pkl)
└── output/                  # Saved evaluation plots and metrics
```

---

## Step 1 — Data Generation (`data_generation.py`)

Generate a synthetic dataset with:

- 10,000 rows
- 20 features: a mix of numeric (continuous + integer) and categorical variables
- Varying degrees of missing data per column (ranging from ~2% to ~95%)
- 1 binary target variable (moderately imbalanced, e.g. 70/30 split)

Save the raw dataset to a path defined in `config.yaml`.

---

## Step 2 — EDA (`eda.py`)

- Load the raw dataset
- Run automated EDA using **Sweetviz** (preferred over ydata-profiling: faster on 10k rows, cleaner target-correlation report, no heavy dependencies)
- Save the HTML report to the `/output` folder

---

## Step 3 — Data Preprocessing (`data_prep.py`)

- Load raw data
- Drop columns where missing data exceeds 90%
- Drop constant or near-constant columns
- Perform train/test split (stratified on target), with split ratio defined in config
- Impute missing values appropriately (median for numeric, mode for categorical)
- Encode categorical variables using **OrdinalEncoder** (appropriate for tree-based models)
- Scale numeric features using **StandardScaler**
  - *Justification: synthetic numeric data is approximately normal with no extreme skew; StandardScaler (zero mean, unit variance) is appropriate. MinMaxScaler would only be preferred if a strict bounded input range were required.*
- Save the processed train/test sets to paths defined in config

---

## Step 4 — Training (`train.py`)

- Load processed train data
- Train a binary classification model using **XGBoost**
- Handle class imbalance using `scale_pos_weight = n_negative / n_positive` (avoids SMOTE artefacts on already-synthetic data)
- **Feature selection via two-pass tree importance** (no RandomizedSearchCV):
  - **Pass 1:** Fit lightweight XGBoost on all features → extract gain-based `feature_importances_` → rank and select top N
  - **Pass 2:** Refit final model on top-N features only
  - *Justification: gain-based importance is model-native, fast, handles mixed types after encoding, and is consistent with the final estimator — no extra estimator needed*
- Save the trained model as `.pkl` and selected feature list as `.pkl` to `/model`

---

## Step 5 — Evaluation (`evaluate.py`)

Load the saved model and test data. Generate and save the following to `/output`:

- Confusion matrix (heatmap with counts + row-normalised %)
- ROC-AUC curve
- Precision-Recall (AUC-PR) curve
- Feature importance bar chart
- A summary metrics file (`metrics_summary.csv`) with: F1 Score, Accuracy, Precision, Recall, ROC-AUC, PR-AUC

---

## Additional Requirements

- All file paths, model parameters, and key thresholds defined in `config/config.yaml` — no hardcoding
- `main.py` runs the full pipeline in sequence by importing and calling functions from each helper script
- Code must be clean, modular, and well-commented
- `README.md` includes: project overview, folder structure, setup instructions (`pip install -r requirements.txt`), and how to run (`python main.py`)
- Include a `requirements.txt`

---

## Key Design Decisions (Agreed During Planning)

| Decision | Choice | Rationale |
|---|---|---|
| Python version | 3.13 (system default) | Used whatever environment was available |
| EDA library | Sweetviz | 5–10× faster than ydata-profiling on 10k rows; no heavy deps; built-in target correlation view |
| Hyperparameter tuning | None (fixed params in config) | Not required; config-driven params are sufficient |
| Feature selection | XGBoost gain-based importance | Model-native, no extra estimator, consistent with final model |
| Class imbalance | `scale_pos_weight` | Avoids synthetic over-sampling artefacts from SMOTE on already-synthetic data |
| Categorical encoding | OrdinalEncoder | Sufficient for tree-based models; avoids one-hot overhead |
| Numeric scaling | StandardScaler | Data is approximately normal; zero-mean/unit-variance scaling is appropriate |

---

## Phased Implementation Plan

| Phase | Steps | Status |
|---|---|---|
| Phase 1 | Folder structure, `config.yaml`, `requirements.txt`, `README.md` | ✅ Complete |
| Phase 2 | `data_generation.py`, `eda.py` | ✅ Complete |
| Phase 3 | `data_prep.py` | ✅ Complete |
| Phase 4 | `train.py` | ✅ Complete |
| Phase 5 | `evaluate.py` | ✅ Complete |
| Phase 6 | `main.py` orchestrator | ✅ Complete |
