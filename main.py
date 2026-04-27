# =============================================================================
# main.py
# Pipeline Orchestrator — runs all five stages end-to-end.
#
# Usage:
#   python main.py
#
# Stages executed in order:
#   1. Data Generation   — helper/data_generation.py
#   2. EDA               — helper/eda.py
#   3. Data Preparation  — helper/data_prep.py
#   4. Model Training    — helper/train.py
#   5. Evaluation        — helper/evaluate.py
#
# All configuration (paths, params, thresholds) is read from config/config.yaml.
# No values are hardcoded here or in any helper module.
# =============================================================================

import logging
import sys
import time

from helper.data_generation import generate_data
from helper.eda import run_eda
from helper.data_prep import prepare_data
from helper.train import train_model
from helper.evaluate import evaluate_model

# ---------------------------------------------------------------------------
# Logging setup — timestamped, INFO level, writes to stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/config.yaml"


# ---------------------------------------------------------------------------
# Stage runner — wraps each step with timing and error handling
# ---------------------------------------------------------------------------

def _run_stage(name: str, fn, *args, **kwargs) -> None:
    """
    Executes a pipeline stage function, logs elapsed time, and re-raises
    any exception with a clear stage label so failures are easy to locate.
    """
    logger.info(f"{'='*60}")
    logger.info(f"  Starting: {name}")
    logger.info(f"{'='*60}")
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.error(f"Stage '{name}' failed: {exc}", exc_info=True)
        raise
    elapsed = time.perf_counter() - t0
    logger.info(f"  Completed: {name}  ({elapsed:.1f}s)")
    logger.info("")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_start = time.perf_counter()
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║          Binary Classification Pipeline                  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info("")

    _run_stage("Step 1 — Data Generation",   generate_data, CONFIG_PATH)
    _run_stage("Step 2 — EDA",               run_eda,       CONFIG_PATH)
    _run_stage("Step 3 — Data Preparation",  prepare_data,  CONFIG_PATH)
    _run_stage("Step 4 — Model Training",    train_model,   CONFIG_PATH)
    _run_stage("Step 5 — Evaluation",        evaluate_model, CONFIG_PATH)

    total = time.perf_counter() - pipeline_start
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  Pipeline complete  ·  total time: {total:.1f}s{' '*(23-len(f'{total:.1f}'))}║")
    logger.info("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
