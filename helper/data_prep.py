# =============================================================================
# data_prep.py
# Step 3 — Data Preprocessing Pipeline.
#
# Sequential steps:
#   1. Load raw data
#   2. Drop columns where missing rate > missing_drop_threshold (default 90%)
#   3. Drop near-constant columns (one value dominates > near_constant_threshold)
#   4. Stratified train / test split
#   5. Impute missing values  — median for numeric, mode for categorical
#   6. Encode categorical features — OrdinalEncoder
#      Justification: downstream model is XGBoost (tree-based); ordinal encoding
#      is sufficient and avoids the high-dimensionality overhead of one-hot
#      encoding for low-cardinality categoricals.
#   7. Scale numeric features — StandardScaler
#      Justification: synthetic numeric data is approximately normal with no
#      extreme bounded range, so zero-mean / unit-variance scaling is appropriate.
#      MinMaxScaler would only be preferred if a strict [0,1] input range were
#      required (e.g. neural nets). Normalization (L2 row-wise) is not applied
#      because features are independent tabular columns, not embedding vectors.
#
# Output:
#   data/train.csv   (path defined in config.yaml)
#   data/test.csv    (path defined in config.yaml)
# =============================================================================

import os
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Step 2 — Drop high-missing columns
# ---------------------------------------------------------------------------

def _drop_high_missing(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop any feature column whose missing rate exceeds `threshold`.
    The target column is always preserved.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    missing_rates = df[feature_cols].isnull().mean()
    cols_to_drop = missing_rates[missing_rates > threshold].index.tolist()

    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} high-missing columns (>{threshold:.0%}): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info(f"No columns exceed the {threshold:.0%} missing threshold — none dropped.")

    return df


# ---------------------------------------------------------------------------
# Step 3 — Drop near-constant columns
# ---------------------------------------------------------------------------

def _drop_near_constant(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop feature columns where a single value accounts for more than
    `threshold` fraction of all non-null values.
    Near-constant columns carry almost no predictive signal.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    cols_to_drop = []

    for col in feature_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            cols_to_drop.append(col)
            continue
        top_freq = non_null.value_counts(normalize=True).iloc[0]
        if top_freq > threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        logger.info(
            f"Dropping {len(cols_to_drop)} near-constant columns (>{threshold:.0%} single value): {cols_to_drop}"
        )
        df = df.drop(columns=cols_to_drop)
    else:
        logger.info(f"No near-constant columns found at threshold {threshold:.0%}.")

    return df


# ---------------------------------------------------------------------------
# Step 4 — Stratified train / test split
# ---------------------------------------------------------------------------

def _split(
    df: pd.DataFrame, test_size: float, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split preserves the target class ratio in both sets.
    Returns (train_df, test_df) both with the target column included.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    logger.info(
        f"Train size: {len(train_df):,}  |  Test size: {len(test_df):,}  "
        f"|  Train positive rate: {y_train.mean():.3f}  "
        f"|  Test positive rate: {y_test.mean():.3f}"
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Step 5-7 — Impute, encode, scale
# ---------------------------------------------------------------------------

def _preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fits all transformers on the TRAINING set only, then applies them to
    both train and test sets — preventing any data leakage from the test set.

    Transformers applied:
      - SimpleImputer(median)   → numeric columns
      - SimpleImputer(mode)     → categorical columns
      - OrdinalEncoder          → categorical columns
      - StandardScaler          → numeric columns
    """
    target_train = train_df["target"].reset_index(drop=True)
    target_test  = test_df["target"].reset_index(drop=True)

    X_train = train_df.drop(columns=["target"]).reset_index(drop=True)
    X_test  = test_df.drop(columns=["target"]).reset_index(drop=True)

    # Identify column types
    numeric_cols     = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(f"Numeric columns    ({len(numeric_cols)}): {numeric_cols}")
    logger.info(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # ---- Numeric pipeline ------------------------------------------------
    if numeric_cols:
        # Impute with median (robust to skewed distributions)
        num_imputer = SimpleImputer(strategy="median")
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = num_imputer.transform(X_test[numeric_cols])

        # Scale: StandardScaler — zero mean, unit variance
        scaler = StandardScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # ---- Categorical pipeline --------------------------------------------
    if categorical_cols:
        # Impute with most frequent value (mode)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols]  = cat_imputer.transform(X_test[categorical_cols])

        # Encode: OrdinalEncoder
        # handle_unknown="use_encoded_value" with unknown_value=-1 gracefully
        # handles any unseen category in the test set without raising an error.
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols]  = encoder.transform(X_test[categorical_cols])

    # Re-attach target
    X_train["target"] = target_train.values
    X_test["target"]  = target_test.values

    return X_train, X_test


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_data(config_path: str = "config/config.yaml") -> None:
    """
    Full preprocessing pipeline.
    Reads raw data, cleans, transforms, and saves processed train/test splits.
    """
    cfg         = _load_config(config_path)
    seed        = cfg["random_seed"]
    raw_path    = cfg["paths"]["raw_data"]
    train_path  = cfg["paths"]["train_data"]
    test_path   = cfg["paths"]["test_data"]
    miss_thresh = cfg["preprocessing"]["missing_drop_threshold"]
    nc_thresh   = cfg["preprocessing"]["near_constant_threshold"]
    test_size   = cfg["preprocessing"]["test_size"]

    logger.info("=== Step 3: Data Preprocessing ===")
    logger.info(f"Loading raw data from {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    # Step 2 — drop high-missing
    df = _drop_high_missing(df, threshold=miss_thresh)

    # Step 3 — drop near-constant
    df = _drop_near_constant(df, threshold=nc_thresh)

    logger.info(f"Shape after column drops: {df.shape}")

    # Step 4 — stratified split
    train_df, test_df = _split(df, test_size=test_size, random_seed=seed)

    # Steps 5-7 — impute, encode, scale (fit on train only)
    train_df, test_df = _preprocess_features(train_df, test_df, random_seed=seed)

    # Save
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Train set saved → {train_path}  shape: {train_df.shape}")
    logger.info(f"Test  set saved → {test_path}   shape: {test_df.shape}")


# ---------------------------------------------------------------------------
# Allow direct execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    prepare_data()
