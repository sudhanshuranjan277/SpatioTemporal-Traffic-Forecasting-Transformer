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

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"


# =============================================================================
# Dataset
# =============================================================================

DATASET_FILES = sorted(PROCESSED_DATA_DIR.glob("*.csv"))

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

HISTORY_LENGTH = 12

PREDICTION_HORIZON = 3

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
    print(f"Prediction Horizon: {PREDICTION_HORIZON}")

    print(f"Target Column     : {TARGET_COLUMN}")

    print(f"Device            : {DEVICE}")

    print("=" * 60)
    
    # Structure Generator

NODE_EMBEDDING_DIM = EMBEDDING_DIM

STRUCTURE_DROPOUT = 0.1