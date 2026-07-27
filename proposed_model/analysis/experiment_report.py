"""
analysis/experiment_report.py

Generate TransGTR Experiment Report

Output
------
outputs/experiment_report.json
"""


from __future__ import annotations


import json


from pathlib import Path


from configs.config import (
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    HISTORY_LENGTH,
    PREDICTION_HORIZON,
    NUM_INPUT_FEATURES,
)





# ==========================================================
# Load Existing Results
# ==========================================================


def load_metrics():

    metrics_file = (
        OUTPUT_DIR /
        "test_metrics.json"
    )


    if not metrics_file.exists():

        raise FileNotFoundError(
            f"Metrics file not found: {metrics_file}"
        )


    with open(
        metrics_file,
        "r"
    ) as file:

        metrics = json.load(file)


    return metrics





def load_training_history():

    history_file = (
        OUTPUT_DIR /
        "training_history.json"
    )


    if not history_file.exists():

        raise FileNotFoundError(
            f"Training history not found: {history_file}"
        )


    with open(
        history_file,
        "r"
    ) as file:

        history = json.load(file)


    return history
