# 🚦 SpatioTemporal Traffic Forecasting Transformer (TransGTR)

## Transformer-based Graph Traffic Forecasting Framework


A research-oriented deep learning framework for **spatio-temporal traffic forecasting** using Transformer-based graph learning.

Inspired by:

> Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities  
> Jin, Chen & Yang  
> KDD 2023


This repository implements a TransGTR-inspired framework focusing on:

- Temporal dependency learning
- Graph-based spatial representation learning
- Multi-step traffic forecasting
- Baseline model comparison
- Reproducible research workflow


---

# 📌 Project Objective


The objective of this project is to develop a scalable traffic forecasting framework capable of predicting future traffic conditions across multiple road network nodes.


The framework focuses on:

- Transformer-based traffic prediction
- Graph-aware feature learning
- Multi-horizon forecasting
- Comparison with existing deep learning approaches
- Research-oriented evaluation pipeline


---

# 🧠 Proposed Framework


The proposed framework combines:


## Temporal Learning

Transformer architecture is used to capture historical traffic dependencies and temporal patterns.


## Spatial Learning

Graph-based representation learning captures relationships between different traffic nodes.


## Multi-step Forecasting

The model predicts future traffic conditions for multiple forecasting horizons.


Overall pipeline:
raffic Dataset

    ↓

Data Loader

    ↓

Sliding Window Generator

    ↓

Feature Tensor Creation

    ↓

TransGTR Model

    ↓

Prediction Head

    ↓

Evaluation Metrics

    ↓

Visualization



---

# 🏗️ Repository Structure



SpatioTemporal-Traffic-Forecasting-Transformer

│
├── proposed_model/
│ │
│ ├── models/
│ │ └── TransGTR architecture
│ │
│ ├── data/
│ │ ├── loader.py
│ │ ├── dataset.py
│ │ ├── window.py
│ │ └── evaluation_dataset.py
│ │
│ ├── trainer/
│ │ └── training pipeline
│ │
│ ├── evaluation/
│ │ └── final result generation
│ │
│ └── visualization/
│
│
├── comparison_models/
│ │
│ ├── lstm/
│ ├── gru/
│ └── graph_wavenet/
│
│
├── datasets/
│
├── outputs/
│
├── README.md
├── architecture.md
├── roadmap.md
└── requirements.txt



---

# 🔄 Complete Forecasting Pipeline



Traffic Data

  ↓

Dataset Validation

  ↓

Sliding Window Generation

  ↓

Historical Traffic Features

  ↓

Graph Transformer Model

  ↓

Future Traffic Prediction

  ↓

Performance Evaluation

  ↓

Result Visualization



---

# 🚀 Implemented Components


## Dataset Pipeline


Implemented:


✅ Dataset loading  
✅ Dataset validation  
✅ Missing value handling  
✅ Sliding window generation  
✅ PyTorch Dataset integration  
✅ Multi-location evaluation dataset  


Current evaluation setup:



Dataset:

Location 1 + Location 2

Total Nodes:

9

Input:

(12, 9, 13)

Forecast:

Multi-step traffic prediction



---

# 🤖 Proposed Model


Implemented:


✅ Transformer-based temporal learning  

✅ Graph-aware traffic representation  

✅ Prediction head  

✅ Training pipeline  

✅ Testing pipeline  

✅ Multi-horizon forecasting  


---

---

# 📊 Baseline Comparison Models


The framework includes multiple deep learning baseline models for performance comparison.


| Model | Type | Status |
|-|-|-|
| TransGTR | Proposed Transformer Graph Model | ✅ Completed |
| LSTM | Recurrent Neural Network Baseline | ✅ Completed |
| GRU | Recurrent Neural Network Baseline | ✅ Completed |
| GraphWaveNet | Graph Neural Network Baseline | ✅ Completed |


The objective is to evaluate whether the proposed TransGTR architecture provides better forecasting capability compared with existing approaches.


---

# 🔮 Multi-Horizon Forecasting


The evaluation pipeline supports multiple future forecasting horizons.


| Forecast Horizon | Forecasting Strategy |
|-|-|
| 3 Minutes | Direct Forecasting |
| 5 Minutes | Direct Forecasting |
| 10 Minutes | Recursive Multi-Step Forecasting |


For longer forecasting horizons, recursive forecasting is applied where previous predictions are used to generate future traffic states.


---

# 📈 Evaluation Metrics


The following metrics are used for model evaluation:


| Metric | Description |
|-|-|
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |


---

# 🏆 Experimental Results


Evaluation was performed using combined traffic datasets:

Location 1 + Location 2

Total Nodes: 9

Forecast Horizons:

3 Minutes
5 Minutes
10 Minutes



---

# ⏱️ 3 Minute Forecasting Results


| Model | MSE | MAE | RMSE | R² |
|-|-|-|-|-|
| TransGTR | 130.247 | 8.098 | 11.413 | 0.515 |
| LSTM | 223.467 | 13.403 | 14.949 | 0.169 |
| GRU | 221.554 | 13.307 | 14.885 | 0.176 |
| GraphWaveNet | 168.107 | 10.517 | 12.966 | 0.375 |


---

# ⏱️ 5 Minute Forecasting Results


| Model | MSE | MAE | RMSE | R² |
|-|-|-|-|-|
| TransGTR | 133.145 | 8.167 | 11.539 | 0.510 |
| LSTM | 223.753 | 13.425 | 14.958 | 0.176 |
| GRU | 223.079 | 13.369 | 14.936 | 0.178 |
| GraphWaveNet | 171.752 | 10.599 | 13.105 | 0.367 |


---

# ⏱️ 10 Minute Forecasting Results


| Model | MSE | MAE | RMSE | R² |
|-|-|-|-|-|
| TransGTR | 186.935 | 10.896 | 13.672 | 0.300 |
| LSTM | 242.159 | 13.722 | 15.561 | 0.093 |
| GRU | 244.614 | 13.744 | 15.640 | 0.083 |
| GraphWaveNet | 347.028 | 14.460 | 18.629 | -0.300 |


---

# 🥇 Overall Performance Analysis


Across all forecasting horizons, TransGTR achieved the best performance.


Key observations:


✅ Lowest prediction error among all models  

✅ Highest R² score across forecasting horizons  

✅ Better long-term forecasting stability  

✅ Effective temporal and spatial traffic representation learning  


Overall ranking:


| Rank | Model |
|-|-|
| 🥇 1 | TransGTR |
| 🥈 2 | GraphWaveNet |
| 🥉 3 | GRU |
| 4 | LSTM |


---

# 📂 Generated Output Files


Final experimental outputs are generated inside:



outputs/

└── final_results/

├── model_comparison.csv

├── model_comparison.xlsx

├── accuracy_comparison.png

├── horizon_rmse_comparison.png

├── mse_comparison.png

├── mae_comparison.png

└── rmse_comparison.png


---

# 🚦 Traffic Impact Evaluation


The framework also evaluates traffic system-level impact:


Generated metrics:



outputs/

└── comparison/

└── traffic_metrics/

    ├── queue_length.csv

    ├── waiting_time.csv

    └── spillback.csv


These metrics are used to analyze:

- Traffic congestion behavior
- Queue formation
- Vehicle waiting time
- Spillback events


---

---

# ⚙️ Installation


Clone the repository:


```bash
git clone https://github.com/sudhanshuranjan277/SpatioTemporal-Traffic-Forecasting-Transformer.git

cd SpatioTemporal-Traffic-Forecasting-Transformer

nstall dependencies:

pip install -r requirements.txt
🚀 Running the Project
Dataset Pipeline Test
python -m proposed_model.data.dataset

Expected:

TrafficDataset Test

Train Samples
Validation Samples
Test Samples

Input Shape
Target Shape
Train Proposed Model

Run TransGTR training:

python -m proposed_model.trainer.trainer
Train Baseline Models

Run baseline experiments:

python -m comparison_models.train_baselines
Run Multi-Horizon Testing

Execute model evaluation:

python -m proposed_model.test

The testing pipeline evaluates:

Models:

TransGTR
LSTM
GRU
GraphWaveNet


Horizons:

3 Minutes
5 Minutes
10 Minutes
Generate Final Research Results

Generate comparison tables and graphs:

python -m proposed_model.evaluation.generate_final_results

Generated files:

outputs/final_results/

model_comparison.csv

model_comparison.xlsx

accuracy_comparison.png

horizon_rmse_comparison.png

mse_comparison.png

mae_comparison.png

rmse_comparison.png
📊 Visualization

The project generates:

Model Accuracy Comparison
accuracy_comparison.png

Shows R² performance comparison between all forecasting models.

Error Analysis

Generated:

mse_comparison.png

mae_comparison.png

rmse_comparison.png

Used for comparing prediction errors.

Horizon Performance Analysis

Generated:

horizon_rmse_comparison.png

Shows how forecasting performance changes with increasing prediction horizon.

🔬 Research Direction

Future improvements:

Dynamic graph structure learning
Multi-city traffic transfer learning
Real-time traffic prediction
Reinforcement learning based traffic signal optimization
Large-scale SUMO simulation experiments
Adaptive traffic control integration
📚 Reference

Jin, Y., Chen, K., & Yang, Q.

Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities

KDD 2023

👤 Author

Sudhanshu Ranjan

Project:

SpatioTemporal Traffic Forecasting Transformer (TransGTR)

✅ Project Completion Status
Component	Status
Repository Setup	✅ Completed
Dataset Pipeline	✅ Completed
TransGTR Implementation	✅ Completed
Baseline Models	✅ Completed
Multi-Horizon Forecasting	✅ Completed
Model Evaluation	✅ Completed
Result Visualization	✅ Completed
Final Documentation	✅ Completed
🎯 Final Summary

This project provides a complete research framework for spatio-temporal traffic forecasting using Transformer-based graph learning.

The final system supports:

✅ Multi-node traffic forecasting

✅ Multi-horizon prediction

✅ Transformer-based learning

✅ Baseline comparison

✅ Research-level evaluation pipeline


---