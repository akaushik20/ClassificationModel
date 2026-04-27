# =============================================================================
# train.py
# Step 4 — Model Training with Tree-Based Feature Selection.
#
# Two-pass approach:
#   Pass 1 — Fit XGBoost on ALL available features to obtain feature
#             importances (gain-based). Rank features by importance and
#             select the top-N defined in config.yaml.
#             Justification: gain-based importance measures the average
#             improvement in loss brought by a feature across all splits.
#             It is model-native (no extra estimator), handles mixed
#             numeric/encoded-categorical inputs, and is consistent with the
#             final estimator — avoiding any leakage from a separate selector.
#
#   Pass 2 — Refit the final XGBoost model using ONLY the selected top-N
#             features. This is the model saved as the production artifact.
#
# Class imbalance:
#   scale_pos_weight = n_negative / n_positive is computed from the training
#   set and passed to XGBoost. This re-weights the loss without generating
#   synthetic samples (SMOTE), which is appropriate here since the data is
#   already synthetic and SMOTE would add no real information.
#
# Output:
#   model/model.pkl     — trained XGBClassifier (top-N features)
#   model/features.pkl  — ordered list of selected feature names
# =============================================================================

import os
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pass 1 — Feature importance ranking
# ---------------------------------------------------------------------------

def _select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int,
    model_params: dict,
    scale_pos_weight: float,
    random_seed: int,
) -> list[str]:
    """
    Fit a lightweight XGBoost on all features to extract gain-based
    feature importances. Returns the names of the top-N features sorted
    by descending importance.

    A smaller n_estimators is used in this pass to keep runtime low —
    the goal is ranking, not optimal performance.
    """
    logger.info(f"Pass 1 — fitting XGBoost on all {X_train.shape[1]} features for importance ranking…")

    selector_model = XGBClassifier(
        n_estimators=100,                      # Lightweight — ranking pass only
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        eval_metric="logloss",
        verbosity=0,
    )
    selector_model.fit(X_train, y_train)

    # Build importance DataFrame
    importance_df = pd.DataFrame({
        "feature":    X_train.columns.tolist(),
        "importance": selector_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info("Feature importances (all features):")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:<25}  {row['importance']:.6f}")

    # Cap top_n at available feature count
    top_n = min(top_n, len(importance_df))
    selected = importance_df.head(top_n)["feature"].tolist()
    logger.info(f"Selected top {top_n} features: {selected}")

    return selected


# ---------------------------------------------------------------------------
# Pass 2 — Final model training on selected features
# ---------------------------------------------------------------------------

def _train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    model_params: dict,
    scale_pos_weight: float,
    random_seed: int,
) -> XGBClassifier:
    """
    Refit the full XGBoost model using only the selected feature subset.
    This is the artifact that will be used for evaluation and inference.
    """
    logger.info(
        f"Pass 2 — fitting final XGBoost on top {len(selected_features)} features "
        f"(n_estimators={model_params['n_estimators']})…"
    )

    final_model = XGBClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        learning_rate=model_params["learning_rate"],
        subsample=model_params["subsample"],
        colsample_bytree=model_params["colsample_bytree"],
        min_child_weight=model_params["min_child_weight"],
        gamma=model_params["gamma"],
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        eval_metric="logloss",
        verbosity=0,
    )
    final_model.fit(X_train[selected_features], y_train)

    logger.info("Final model training complete.")
    return final_model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_model(config_path: str = "config/config.yaml") -> None:
    """
    Full training pipeline:
      1. Load processed training data
      2. Compute scale_pos_weight from class distribution
      3. Pass 1 — fit on all features, extract top-N by gain importance
      4. Pass 2 — refit final model on selected features only
      5. Save model.pkl and features.pkl to /model
    """
    cfg           = _load_config(config_path)
    seed          = cfg["random_seed"]
    train_path    = cfg["paths"]["train_data"]
    model_path    = cfg["paths"]["model"]
    features_path = cfg["paths"]["features"]
    top_n         = cfg["feature_selection"]["top_n_features"]
    model_params  = cfg["model"]
    use_spw       = model_params.get("use_scale_pos_weight", True)

    logger.info("=== Step 4: Model Training ===")
    logger.info(f"Loading training data from {train_path}")

    train_df = pd.read_csv(train_path)
    X_train  = train_df.drop(columns=["target"])
    y_train  = train_df["target"]

    logger.info(f"Training set shape: {X_train.shape}")

    # Compute scale_pos_weight for class imbalance handling
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if use_spw else 1.0
    logger.info(
        f"Class counts — negative: {n_neg:,}  positive: {n_pos:,}  "
        f"scale_pos_weight: {scale_pos_weight:.4f}"
    )

    # Pass 1 — feature selection via importance ranking
    selected_features = _select_top_features(
        X_train, y_train, top_n, model_params, scale_pos_weight, seed
    )

    # Pass 2 — final model on reduced feature set
    final_model = _train_final_model(
        X_train, y_train, selected_features, model_params, scale_pos_weight, seed
    )

    # Save artifacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    joblib.dump(selected_features, features_path)

    logger.info(f"Model saved       → {model_path}")
    logger.info(f"Feature list saved → {features_path}")


# ---------------------------------------------------------------------------
# Allow direct execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    train_model()
