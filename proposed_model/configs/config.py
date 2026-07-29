"""
Project: Traffic Forecasting Research
Model  : TransGTR (From Scratch)

Central configuration file.

NOTE:
- No hardcoded values should appear anywhere else in the project.
- Every module must import configuration from this file.
"""

from pathlib import Path

import torch


# =============================================================================
# Project Directories
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT.parent / "datasets"

RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"


# =============================================================================
# Output Directories
# =============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"


CHECKPOINT_DIR = (
    OUTPUT_DIR
    /
    "checkpoints"
)


LOG_DIR = (
    OUTPUT_DIR
    /
    "logs"
)


# =============================================================================
# Model Checkpoint Directories
# =============================================================================

TRANSGTR_CHECKPOINT_DIR = (
    CHECKPOINT_DIR
    /
    "transgtr"
)


BASELINE_CHECKPOINT_DIR = (
    CHECKPOINT_DIR
    /
    "baselines"
)



# =============================================================================
# Result Directories
# =============================================================================

METRIC_DIR = (
    OUTPUT_DIR
    /
    "metrics"
)


PREDICTION_DIR = (
    OUTPUT_DIR
    /
    "predictions"
)


GRAPH_DIR = (
    OUTPUT_DIR
    /
    "graphs"
)


REPORT_DIR = (
    OUTPUT_DIR
    /
    "reports"
)


TRAFFIC_METRIC_DIR = (
    OUTPUT_DIR
    /
    "traffic_metrics"
)



# =============================================================================
# Experiment Output Directories
# =============================================================================

COMPARISON_OUTPUT_DIR = (
    OUTPUT_DIR
    /
    "comparison"
)


HORIZON_3_DIR = (
    COMPARISON_OUTPUT_DIR
    /
    "horizon_3min"
)


HORIZON_5_DIR = (
    COMPARISON_OUTPUT_DIR
    /
    "horizon_5min"
)


HORIZON_10_DIR = (
    COMPARISON_OUTPUT_DIR
    /
    "horizon_10min"
)



# =============================================================================
# Dataset
# =============================================================================

DATASET_FILES = sorted(
    PROCESSED_DATA_DIR.glob("*.csv")
)


TIMESTAMP_COLUMN = "simulation_time"

LOCATION_COLUMN = "location_id"

NODE_COLUMN = "junction_id"



# =============================================================================
# Feature Configuration
# =============================================================================

FEATURE_COLUMNS = [

    "vehicle_count",

    "traffic_flow",

    "arrival_rate",

    "departure_rate",

    "traffic_event_type",

    "remaining_green_time",

    "current_signal_phase",

    "downstream_occupancy",

    "downstream_queue_length",

    "average_speed",

    "waiting_time",

    "travel_time",

    "queue_length",

]


TARGET_COLUMN = "traffic_flow"

# =============================================================================
# Window Configuration
# =============================================================================
# Historical observation window
# Example:
# Last 12 minutes traffic data is used for prediction

HISTORY_LENGTH = 12


# Multi-step forecasting experiments

FORECAST_HORIZONS = [
    3,      # 3 minutes ahead
    5,      # 5 minutes ahead
    10      # 10 minutes ahead
]


# Default horizon
# Used by normal training/testing

PREDICTION_HORIZON = 5


SLIDING_WINDOW_STRIDE = 1

# ==========================
# Input Features
# ==========================

FEATURE_COLUMNS = [
    "vehicle_count",
    "traffic_flow",
    "arrival_rate",
    "departure_rate",
    "traffic_event_type",
    "remaining_green_time",
    "current_signal_phase",
    "downstream_occupancy",
    "downstream_queue_length",
    "average_speed",
    "waiting_time",
    "travel_time",
    "queue_length",
]

NUM_INPUT_FEATURES = len(FEATURE_COLUMNS)

# ==========================
# Model Parameters
# ==========================

EMBEDDING_DIM = 64


# =============================================================================
# Dataset Split
# =============================================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.10

TEST_RATIO = 0.20


# =============================================================================
# Training
# =============================================================================

BATCH_SIZE = 64

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

# ==========================
# Transformer Parameters
# ==========================

NUM_HEADS = 8
NUM_LAYERS = 4
HIDDEN_DIM = 256
DROPOUT = 0.1




# =============================================================================
# Device
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42


# =============================================================================
# Validation
# =============================================================================

REQUIRED_COLUMNS = [

    LOCATION_COLUMN,
    NODE_COLUMN,
    TIMESTAMP_COLUMN,

    *FEATURE_COLUMNS,

]

# ==========================================================
# SCALER CONFIGURATION
# ==========================================================

SCALER_TYPE = "standard"      # standard | minmax
SCALER_SAVE_NAME = "feature_scaler.pkl"

# =============================================================================
# Utility
# =============================================================================

def show_config() -> None:
    """Print important project configuration."""

    print("=" * 60)
    print("Traffic Forecasting Research Configuration")
    print("=" * 60)

    print(f"Project Root      : {PROJECT_ROOT}")
    print(f"Processed Dataset : {PROCESSED_DATA_DIR}")
    print(f"Datasets Found    : {len(DATASET_FILES)}")

    print(f"History Length    : {HISTORY_LENGTH}")
    print(f"Prediction Horizon: {FORECAST_HORIZON}")

    print(f"Target Column     : {TARGET_COLUMN}")

    print(f"Device            : {DEVICE}")

    print("=" * 60)
    
    # Structure Generator

NODE_EMBEDDING_DIM = EMBEDDING_DIM

STRUCTURE_DROPOUT = 0.1