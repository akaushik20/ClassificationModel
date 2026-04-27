# =============================================================================
# eda.py
# Step 2 — Automated Exploratory Data Analysis.
#
# Loads the raw dataset and generates a Sweetviz HTML report.
#
# Why Sweetviz over ydata-profiling?
#   - 5-10x faster on 10k rows — no heavy optional dependency chain
#   - Built-in target-correlation view ideal for binary classification tasks
#   - Single self-contained HTML output, easy to share with stakeholders
#
# Output: output/eda_report.html   (path defined in config.yaml)
# =============================================================================

import os
import logging

import pandas as pd
import sweetviz as sv
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — load config
# ---------------------------------------------------------------------------

def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eda(config_path: str = "config/config.yaml") -> None:
    """
    Runs Sweetviz EDA on the raw dataset and saves an HTML report.

    Sweetviz automatically:
      - Profiles every feature (distribution, missing %, type)
      - Highlights associations with the binary target column
      - Flags high-missing and low-variance columns visually

    The report is opened in the browser when run interactively. When called
    from main.py (non-interactive), show_html() writes the file without
    launching the browser (open_browser=False).
    """
    cfg         = _load_config(config_path)
    raw_path    = cfg["paths"]["raw_data"]
    report_path = cfg["paths"]["eda_report"]

    logger.info("=== Step 2: EDA ===")
    logger.info(f"Loading raw data from {raw_path}")

    df = pd.read_csv(raw_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # ---------------------------------------------------------------------------
    # Sweetviz report
    # target_feat="target" enables the target-correlation analysis panel,
    # showing which features are most associated with the binary outcome.
    # ---------------------------------------------------------------------------
    logger.info("Generating Sweetviz report (this may take ~15-30 seconds)…")
    report = sv.analyze(
        source=df,
        target_feat="target",   # Binary target — drives association analysis
    )

    # Save HTML without auto-opening the browser (pipeline-safe)
    report.show_html(
        filepath=report_path,
        open_browser=False,
        layout="widescreen",
        scale=1.0,
    )

    logger.info(f"EDA report saved → {report_path}")


# ---------------------------------------------------------------------------
# Allow direct execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    run_eda()
