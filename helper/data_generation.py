# =============================================================================
# data_generation.py
# Step 1 — Synthetic dataset generation.
#
# Generates a 10,000-row dataset with:
#   - 8  continuous numeric features  (float, drawn from normal/skewed dists)
#   - 6  integer numeric features     (counts / ordinal-style integers)
#   - 6  categorical features         (low-cardinality string labels)
#   - 1  binary target                (70/30 class imbalance, minority = 1)
#
# Missing values are injected column-by-column with rates sampled uniformly
# from [missing_rate_min, missing_rate_max] defined in config.yaml.
#
# Output: data/raw.csv   (path defined in config.yaml)
# =============================================================================

import os
import logging

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _make_continuous_features(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create float continuous features with varied distributions:
      - standard normal
      - right-skewed (log-normal)
      - uniform
    Cycling through the three distributions across columns keeps the dataset
    realistic and exercises the scaler's handling of different spreads.
    """
    cols = {}
    dists = ["normal", "lognormal", "uniform"]
    for i in range(n_cols):
        dist = dists[i % len(dists)]
        col_name = f"num_cont_{i+1}"
        if dist == "normal":
            cols[col_name] = rng.normal(loc=0.0, scale=1.0, size=n_rows)
        elif dist == "lognormal":
            cols[col_name] = rng.lognormal(mean=0.0, sigma=0.8, size=n_rows)
        else:  # uniform
            cols[col_name] = rng.uniform(low=-3.0, high=3.0, size=n_rows)
    return pd.DataFrame(cols)


def _make_integer_features(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create integer features mimicking counts or ordinal values:
      - Poisson-distributed counts (e.g. number of events)
      - Bounded integers drawn from a discrete uniform range
    """
    cols = {}
    for i in range(n_cols):
        col_name = f"num_int_{i+1}"
        if i % 2 == 0:
            # Poisson count, lambda cycles between low and moderate values
            lam = [2, 5, 10][i % 3]
            cols[col_name] = rng.poisson(lam=lam, size=n_rows).astype(int)
        else:
            # Discrete uniform between 0 and a small upper bound
            high = [3, 7, 15][i % 3]
            cols[col_name] = rng.integers(low=0, high=high + 1, size=n_rows)
    return pd.DataFrame(cols)


def _make_categorical_features(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Create low-cardinality categorical string features with varying number of
    unique levels (2–5 levels per column) and unequal class probabilities.
    This mimics real-world categoricals such as region, product type, etc.
    """
    category_pools = [
        ["A", "B"],
        ["low", "medium", "high"],
        ["red", "green", "blue", "yellow"],
        ["cat_X", "cat_Y", "cat_Z", "cat_W", "cat_V"],
        ["alpha", "beta", "gamma"],
        ["yes", "no"],
    ]
    cols = {}
    for i in range(n_cols):
        pool = category_pools[i % len(category_pools)]
        # Skew probabilities so categories are not equally likely
        raw_weights = rng.dirichlet(alpha=np.ones(len(pool)) * 0.7)
        col_name = f"cat_{i+1}"
        cols[col_name] = rng.choice(pool, size=n_rows, p=raw_weights)
    return pd.DataFrame(cols)


def _make_target(n_rows: int, positive_frac: float, rng: np.random.Generator) -> pd.Series:
    """
    Binary target with minority-class fraction = positive_frac (e.g. 0.30).
    Uses random draw to avoid perfect block structure.
    """
    target = (rng.random(size=n_rows) < positive_frac).astype(int)
    return pd.Series(target, name="target")


# ---------------------------------------------------------------------------
# Missing value injection
# ---------------------------------------------------------------------------

def _inject_missing(
    df: pd.DataFrame,
    missing_rate_min: float,
    missing_rate_max: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject NaN into each feature column independently.
    Missing rate for each column is sampled uniformly from
    [missing_rate_min, missing_rate_max].

    The target column is intentionally excluded — targets must be complete
    for stratified splitting later.

    At least one column is pushed close to missing_rate_max to ensure the
    high-missing-drop logic in data_prep.py has something to act on.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != "target"]
    n_rows = len(df)

    # Sample a missing rate per column
    rates = rng.uniform(low=missing_rate_min, high=missing_rate_max, size=len(feature_cols))

    # Force at least 2 columns to have >90% missing so the drop step is exercised
    high_missing_indices = rng.choice(len(feature_cols), size=2, replace=False)
    rates[high_missing_indices] = rng.uniform(low=0.91, high=0.97, size=2)

    for col, rate in zip(feature_cols, rates):
        missing_mask = rng.random(size=n_rows) < rate
        df.loc[missing_mask, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_data(config_path: str = "config/config.yaml") -> None:
    """
    Full data generation pipeline.
    Reads parameters from config.yaml, generates the synthetic dataset,
    injects missing values, and saves the result to the raw_data path.
    """
    cfg = _load_config(config_path)
    seed          = cfg["random_seed"]
    dg            = cfg["data_generation"]
    raw_path      = cfg["paths"]["raw_data"]

    n_rows        = dg["n_rows"]
    n_cont        = dg["n_numeric_continuous"]
    n_int         = dg["n_numeric_integer"]
    n_cat         = dg["n_categorical"]
    pos_frac      = dg["target_imbalance"]
    miss_min      = dg["missing_rate_min"]
    miss_max      = dg["missing_rate_max"]

    logger.info("=== Step 1: Data Generation ===")
    logger.info(f"Rows: {n_rows} | Continuous: {n_cont} | Integer: {n_int} | Categorical: {n_cat}")

    rng = np.random.default_rng(seed)

    # Build feature blocks
    df_cont = _make_continuous_features(n_rows, n_cont, rng)
    df_int  = _make_integer_features(n_rows, n_int, rng)
    df_cat  = _make_categorical_features(n_rows, n_cat, rng)
    target  = _make_target(n_rows, pos_frac, rng)

    # Combine into a single DataFrame
    df = pd.concat([df_cont, df_int, df_cat, target], axis=1)

    # Inject missing values (target column is preserved)
    df = _inject_missing(df, miss_min, miss_max, rng)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    df.to_csv(raw_path, index=False)

    # Log class distribution
    counts = df["target"].value_counts()
    logger.info(
        f"Target distribution — class 0: {counts.get(0, 0)} "
        f"({counts.get(0, 0)/n_rows:.1%}), "
        f"class 1: {counts.get(1, 0)} "
        f"({counts.get(1, 0)/n_rows:.1%})"
    )
    logger.info(f"Raw dataset saved → {raw_path}  shape: {df.shape}")


# ---------------------------------------------------------------------------
# Allow direct execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    generate_data()
