# =============================================================================
# evaluate.py
# Step 5 — Model Evaluation & Visualization.
#
# Loads the saved model and selected feature list, aligns the test set to
# those features, and produces the following artifacts in /output:
#
#   confusion_matrix.png    — heatmap with raw counts and percentages
#   roc_curve.png           — ROC curve with AUC annotation
#   pr_curve.png            — Precision-Recall curve with AUC annotation
#   feature_importance.png  — horizontal bar chart of final model importances
#   metrics_summary.csv     — F1, Accuracy, Precision, Recall, ROC-AUC, PR-AUC
# =============================================================================

import os
import logging

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Global plot style — clean, presentation-ready
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helper — load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plot 1 — Confusion Matrix
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """
    Heatmap showing raw counts in each cell plus row-normalised percentages
    as annotations, so both absolute errors and relative rates are visible.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontweight="bold", pad=12)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Plot 2 — ROC-AUC Curve
# ---------------------------------------------------------------------------

def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> float:
    """
    Plots the ROC curve with the AUC score annotated on the chart.
    Returns the ROC-AUC value for the metrics summary.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"ROC curve saved → {save_path}")
    return roc_auc


# ---------------------------------------------------------------------------
# Plot 3 — Precision-Recall Curve
# ---------------------------------------------------------------------------

def _plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> float:
    """
    Plots the Precision-Recall curve.
    The baseline (random classifier) is shown as the positive class prevalence.
    Returns PR-AUC for the metrics summary.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=baseline, color="grey", lw=1, linestyle="--", label=f"Baseline = {baseline:.3f}")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"PR curve saved → {save_path}")
    return pr_auc


# ---------------------------------------------------------------------------
# Plot 4 — Feature Importance
# ---------------------------------------------------------------------------

def _plot_feature_importance(
    model,
    selected_features: list,
    save_path: str,
) -> None:
    """
    Horizontal bar chart of the final model's gain-based feature importances,
    sorted descending so the most important feature is at the top.
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    selected_features,
        "importance": importances,
    }).sort_values("importance", ascending=True)  # ascending for horizontal bar readability

    fig, ax = plt.subplots(figsize=(7, max(4, len(selected_features) * 0.35)))
    bars = ax.barh(importance_df["feature"], importance_df["importance"], color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top Feature Importances — Final Model", fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Annotate values on bars
    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(
            bar.get_width() + importance_df["importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance chart saved → {save_path}")


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def _compute_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    pr_auc: float,
    save_path: str,
) -> None:
    """
    Computes classification metrics and saves them to a CSV file.
    Also prints a formatted table to the console / log.
    """
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1 Score":  f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc,
        "PR-AUC":    pr_auc,
    }

    metrics_df = pd.DataFrame(
        list(metrics.items()), columns=["Metric", "Value"]
    )
    metrics_df["Value"] = metrics_df["Value"].round(6)

    metrics_df.to_csv(save_path, index=False)

    # Pretty-print to log
    logger.info("─" * 35)
    logger.info("  Evaluation Metrics Summary")
    logger.info("─" * 35)
    for _, row in metrics_df.iterrows():
        logger.info(f"  {row['Metric']:<12}  {row['Value']:.4f}")
    logger.info("─" * 35)
    logger.info(f"Metrics summary saved → {save_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_model(config_path: str = "config/config.yaml") -> None:
    """
    Full evaluation pipeline:
      1. Load model, selected features, and test data
      2. Align test set to selected features (prevents column mismatch)
      3. Generate predictions at the configured threshold
      4. Produce and save all four plots
      5. Compute and save the metrics summary CSV
    """
    cfg           = _load_config(config_path)
    model_path    = cfg["paths"]["model"]
    features_path = cfg["paths"]["features"]
    test_path     = cfg["paths"]["test_data"]
    threshold     = cfg["evaluation"]["classification_threshold"]
    out           = cfg["paths"]

    logger.info("=== Step 5: Evaluation ===")

    # Load artifacts
    model             = joblib.load(model_path)
    selected_features = joblib.load(features_path)
    test_df           = pd.read_csv(test_path)

    logger.info(f"Model loaded from       {model_path}")
    logger.info(f"Features loaded from    {features_path}  ({len(selected_features)} features)")
    logger.info(f"Test set loaded from    {test_path}  shape: {test_df.shape}")

    X_test = test_df[selected_features]   # Align to training feature set
    y_true = test_df["target"].values

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    logger.info(
        f"Predictions generated at threshold={threshold}  "
        f"| Positive predicted: {y_pred.sum():,} / {len(y_pred):,}"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out["confusion_matrix"]), exist_ok=True)

    # Plots
    _plot_confusion_matrix(y_true, y_pred, out["confusion_matrix"])
    roc_auc = _plot_roc_curve(y_true, y_prob, out["roc_curve"])
    pr_auc  = _plot_pr_curve(y_true, y_prob, out["pr_curve"])
    _plot_feature_importance(model, selected_features, out["feature_importance"])

    # Metrics
    _compute_and_save_metrics(
        y_true, y_pred, y_prob, roc_auc, pr_auc, out["metrics_summary"]
    )


# ---------------------------------------------------------------------------
# Allow direct execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    evaluate_model()
