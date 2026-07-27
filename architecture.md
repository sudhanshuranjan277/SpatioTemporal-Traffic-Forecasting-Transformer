# Architecture.md

# Traffic Forecasting Research
### System Architecture & Development Guide

---

# 1. Project Goal

Build a scalable traffic forecasting framework using Graph Neural Networks and
Temporal Transformers.

The project contains

- Baseline Models
- Our Proposed Model
- Training Pipeline
- Evaluation Pipeline
- Inference Pipeline

This repository is intended for experimentation, benchmarking and reproducible
research.

---

# 2. Repository explaination

This repository follows one important rule:

> Every module has exactly one responsibility.

No duplicate logic.

No hidden dependencies.

No copy-paste implementation.

Everything must be modular.

---

# 3. High Level Architecture

Dataset
    │
    ▼
Preprocessing
    │
    ▼
Graph Construction
    │
    ▼
Temporal Encoding
    │
    ▼
Forecasting Model
    │
    ▼
Loss Functions
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
Inference

---

# 4. Folder Structure

Traffic-Forecasting-Research/

baseline_models/
proposed_model/
datasets/
configs/
outputs/
scripts/
utils/
experiments/
tests/

README.md
requirements.txt
run.py

---

# 5. Baseline Models

Purpose

Provide benchmark implementations.

These models MUST NEVER contain our research modifications.

Current baseline:

- TransGTR (Reference)

Future baselines:

- LSTM
- GCN
- Graph WaveNet
- STGCN
- DCRNN

---

# 6. Proposed Model

This folder contains ONLY our implementation.

Nothing from baseline models should be directly modified here.

Structure

proposed_model/

preprocessing/
graph/
layers/
models/
losses/
trainer/
evaluation/
inference/

---

# 7. Dataset Pipeline

Raw Dataset

↓

Cleaning

↓

Missing Value Handling

↓

Normalization

↓

Sliding Window Generation

↓

Graph Construction

↓

Training Dataset

Every preprocessing step must be reproducible.

---

# 8. Graph Pipeline

Traffic CSV

+

Adjacency Matrix

↓

Graph Builder

↓

Graph Tensor

↓

Model

---

# 9. Model Pipeline

Input Sequence

↓

Temporal Encoder

↓

Spatial Encoder

↓

Forecast Head

↓

Prediction

---

# 10. Training Pipeline

Load Dataset

↓

Create Graph

↓

Forward Pass

↓

Loss Calculation

↓

Backward Pass

↓

Optimizer Step

↓

Checkpoint Saving

---

# 11. Evaluation Pipeline

Load Checkpoint

↓

Prediction

↓

Metrics

↓

Visualization

↓

Save Results

---

# 12. Inference Pipeline

Load Model

↓

Load Latest Traffic Data

↓

Forecast Future Traffic

↓

Save Predictions

---

# 13. Config Policy

Never hardcode

- dataset paths
- epochs
- learning rate
- hidden dimension
- sequence length

Everything must come from config files.

---

# 14. Coding Standards

One class

↓

One responsibility

One file

↓

One major component

Functions should remain short.

Avoid giant files.

---

# 15. Naming Convention

Classes

PascalCase

Example

TrafficDataset

Functions

snake_case

Example

load_dataset()

Variables

snake_case

Constants

UPPER_CASE

---

# 16. Import Rule

Good

from proposed_model.layers.graph_conv import GraphConv

Bad

from utils import *

---

# 17. Experiments

Every experiment gets its own folder.

Example

experiments/

exp01/

exp02/

exp03/

Each experiment stores

- configuration
- metrics
- plots
- observations

---

# 18. Outputs

Generated automatically.

Never edit manually.

Contains

- checkpoints
- logs
- metrics
- predictions

---

# 19. Development Order

Phase 1

Dataset

Phase 2

Graph Construction

Phase 3

Baseline Validation

Phase 4

Proposed Model

Phase 5

Training

Phase 6

Evaluation

Phase 7

Inference

---

# 20. Golden Rules

Never edit baseline models.

Never duplicate code.

Keep modules independent.

Always document new components.

Keep folder responsibilities clear.

Every experiment must be reproducible.

---

# 21. Long-Term Vision

This repository should evolve into a complete traffic forecasting framework that

- supports multiple datasets
- supports multiple baseline models
- supports custom research models
- allows reproducible experiments
- is suitable for publication-quality benchmarking

The architecture should remain stable while individual modules evolve independently.
