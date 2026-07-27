# 🚦 Traffic Forecasting Research
### TransGTR (KDD 2023) Implementation & Extension Framework

A modular deep learning framework for **graph-based traffic forecasting** inspired by the paper:

> **Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities (TransGTR)**  
> Jin, Chen & Yang  
> KDD 2023

This repository aims to implement the TransGTR framework from scratch with a clean, modular architecture while serving as a foundation for future algorithmic improvements and benchmarking.

---

# Project Objective

The primary goal of this repository is to:

- Implement the TransGTR framework from scratch
- Maintain a modular and scalable codebase
- Reproduce the methodology of the original paper
- Support reproducible experiments
- Provide a clean foundation for developing and evaluating future traffic forecasting algorithms

Rather than directly copying an existing implementation, this project focuses on understanding the methodology and building an independent implementation with clear software engineering principles.

---

# Repository Structure

```text
Traffic-Forecasting-Research/

├── baseline_models/          # Baseline forecasting models
├── proposed_model/           # Future custom algorithm
├── datasets/                 # Dataset loading & preprocessing
├── configs/                  # Configuration files
├── outputs/                  # Logs, checkpoints, metrics
├── scripts/                  # Training & utility scripts
├── experiments/              # Experiment configurations
├── tests/                    # Unit tests
├── utils/                    # Shared utilities
├── docs/                     # Documentation

├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── requirements.txt
└── run.py
```

---

# Project Architecture

The repository is organized into independent modules to keep the implementation maintainable and extensible.

Major components include:

- Dataset Pipeline
- Graph Construction
- Node Feature Learning
- Structure Generator
- Forecasting Model
- Training Pipeline
- Evaluation Pipeline
- Experiment Management

Detailed design decisions are documented in:

```
ARCHITECTURE.md
```

---

# TransGTR Framework Overview

The implementation follows the methodology proposed in the original paper.

```
Traffic Data
      │
      ▼
Node Feature Network (TSFormer)
      │
      ▼
Structure Generator
      │
      ▼
Learned Graph
      │
      ▼
Forecasting Model (Graph WaveNet)
      │
      ▼
Traffic Prediction
```

The complete training pipeline consists of four stages:

```
Node Feature Pretraining
          │
          ▼
Knowledge Distillation
          │
          ▼
Joint Structure Generator +
Forecasting Model Training
          │
          ▼
Target City Fine-tuning
```

---

# Current Implementation

Current implementation includes:

- TSFormer Node Feature Network
- Graph Structure Generator
- Graph WaveNet Forecasting Model
- Knowledge Distillation
- Temporal Decoupled Regularization
- Training Pipeline
- Evaluation Pipeline
- Statistical Analysis

---

# File Organization

| File | Location |
|------|----------|
| config.py | Project Root |
| tsformer.py | prediction/ |
| structure_generator.py | prediction/ |
| gwn_model.py | prediction/ |
| decoupled_regularization.py | prediction/ |
| graph_preprocessing.py | prediction/ |
| transgtr_data.py | prediction/ |
| build_adjacency_matrix.py | scripts/ |
| train_transgtr.py | scripts/ |

---

# Dataset Preparation

This project uses two traffic datasets.

| Dataset | Purpose |
|----------|----------|
| location_2 | Source City (Rich Traffic Data) |
| location_1 | Target City (Limited Traffic Data) |

Before training, adjacency matrices must be generated for both cities.

Example:

```bash
python scripts/build_adjacency_matrix.py \
    --net-file maps/osm/osm.net.xml \
    --dataset data/processed/location_2_dataset.csv \
    --output models/location_2_adjacency.npy

python scripts/build_adjacency_matrix.py \
    --net-file maps/osm/osm.net.xml \
    --dataset data/processed/location_1_dataset.csv \
    --output models/location_1_adjacency.npy
```

If warnings indicate missing junctions, verify that the SUMO network (`.net.xml`) matches the dataset before proceeding.

---

# Training

Run the complete TransGTR training pipeline:

```bash
python scripts/train_transgtr.py
```

The training process performs:

1. Node Feature Pretraining
2. Knowledge Distillation
3. Joint Structure Generator Training
4. Forecasting Model Training
5. Target Fine-tuning
6. Evaluation

---

# Output Files

Training generates:

```
models/
    transgtr_model.pth
    transgtr_scalers.pkl

outputs/
    metrics/
        transgtr_metrics.csv
```

---

# Configuration

All configurable parameters are maintained inside:

```
config.py
```

Configuration includes:

- Dataset settings
- Training parameters
- Model hyperparameters
- Graph settings
- TransGTR-specific parameters

---

# Implementation Notes

This implementation follows the original paper while introducing several practical adaptations for SUMO-generated datasets.

Current simplifications include:

- Smaller hidden dimensions to avoid overfitting on limited simulation data.
- Fixed graph structure learning without Graph WaveNet adaptive adjacency.
- Approximation of temporal features for the decoupling module.
- Simulation-time based day-of-week encoding.

These modifications improve compatibility with simulated traffic data while preserving the core methodology of TransGTR.

---

# Development Philosophy

The repository follows these principles:

- Modular Design
- Single Responsibility Modules
- Reproducible Experiments
- Config-Driven Implementation
- Minimal Hidden Dependencies
- Independent Components
- Clean Documentation

---

# Roadmap

Development progress is tracked in:

```
ROADMAP.md
```

Future work includes:

- Complete TransGTR implementation
- Performance optimization
- Additional benchmark experiments
- Custom forecasting algorithm
- Comparative evaluation against the TransGTR baseline

---

# Documentation

Additional documentation:

```
README.md
ARCHITECTURE.md
ROADMAP.md
```

---

# Reference

**Jin, Y., Chen, K., & Yang, Q.**

**Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities**

Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2023).

---

# License

This repository is intended for academic research and educational purposes.