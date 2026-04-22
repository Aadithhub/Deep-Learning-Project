# Deep Learning-Based Predictive Maintenance Framework for Turbofan Engine RUL Prediction


## Overview

This project develops a deep learning pipeline for **Remaining Useful Life (RUL) prediction** of turbofan engines using the NASA C-MAPSS dataset, combined with a **data-driven predictive maintenance decision framework**. Three deep learning architectures — CNN, LSTM, and CNN+BiLSTM — are evaluated on two sub-datasets (FD001 and FD002) under both all-sensor and filtered-sensor configurations as part of an ablation study. Maintenance decisions are generated using a Random Forest classifier trained on temporal sensor degradation statistics, replacing traditional static rule-based threshold approaches.

---

## Directory Structure

```
├── Code/
│   └── DL_end_sem_Project.ipynb       # Main notebook — full pipeline
├── Data/
│   ├── train_FD001.txt                # FD001 training data (sample)
│   ├── test_FD001.txt                 # FD001 test data (sample)
│   ├── RUL_FD001.txt                  # FD001 ground truth RUL
│   ├── train_FD002.txt                # FD002 training data (sample)
│   ├── test_FD002.txt                 # FD002 test data (sample)
│   └── RUL_FD002.txt                  # FD002 ground truth RUL
├── Results/
│   └── Deep_learning_project_results.pdf  # Problem statement, methodology & results
├── README.md
└── requirements.txt
```

> **Note:** The `Data/` folder contains **sample data only**, not the full NASA C-MAPSS dataset. To reproduce full results, download the complete dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and replace the sample files.

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**

| Property | FD001 | FD002 |
|---|---|---|
| Training Engines | 100 | 260 |
| Test Engines | 100 | 259 |
| Operating Conditions | 1 | 6 |
| Fault Modes | 1 | 1 |
| Sensors | 21 | 21 |

- Each engine record contains 21 sensor readings, 3 operational settings, and cycle number
- RUL is capped at 130 cycles (piecewise linear degradation assumption)
- Sliding window of 30 cycles used for sequence generation

---

## Methodology

### 1. Preprocessing
- RUL label generation and clipping at 130 cycles
- StandardScaler normalisation fitted on training data only

### 2. Sensor Ablation Study
- Variance-based sensor analysis to identify low-variance (non-informative) sensors
- FD001: 7 sensors removed — `sensor_1, 5, 6, 10, 16, 18, 19` → 14 features retained
- FD002: 3 sensors removed — `sensor_10, 15, 16` → 18 features retained
- All models trained on both **all-sensor** and **filtered-sensor** configurations for comparison

### 3. Model Architectures

| Model | Key Layers |
|---|---|
| CNN | 2× Conv1D (64→32 filters) + BatchNorm + MaxPool + Dense(64) |
| LSTM | LSTM(64) + Dense(64) |
| CNN+BiLSTM | 2× Conv1D (64→32) + BatchNorm + Bidirectional LSTM(64) + Dense(64) |

- Window size: 30 cycles
- Train/Validation split: 80/20 (`random_state=42`)
- Loss: MSE (CNN, LSTM) | Huber (CNN+BiLSTM on FD002)
- Optimiser: Adam | Early Stopping | ReduceLROnPlateau (CNN+BiLSTM)

### 4. Data-Driven Maintenance Decision Framework
- Best model predictions (LSTM All Sensors, FD001) fed into a Random Forest classifier
- Feature matrix: temporal statistics (mean, std, min, max) across 30-cycle sensor window + operating condition settings = 87 features
- Maintenance label: predicted RUL ≤ 30 cycles → maintenance required
- Dual-threshold probability output: maintenance at ≥ 0.50, spare parts ordering at ≥ 0.35

---

## Results

### RUL Prediction — Ablation Study

**FD001**

| Model | Sensor Config | RMSE | MAE | PHM Score |
|---|---|---|---|---|
| CNN | All Sensors | 19.671 | 15.212 | 521.88 |
| CNN | Filtered | 22.148 | 17.063 | 788.71 |
| **LSTM** | **All Sensors** | **16.125** | **12.123** | **516.24** |
| LSTM | Filtered | 16.134 | 12.166 | 558.31 |
| CNN+BiLSTM | All Sensors | 19.667 | 14.251 | 1389.12 |
| CNN+BiLSTM | Filtered | 19.205 | 14.405 | 1166.86 |

**FD002**

| Model | Sensor Config | RMSE | MAE | PHM Score |
|---|---|---|---|---|
| CNN | All Sensors | 30.498 | 21.085 | 17510.06 |
| CNN | Filtered | 32.585 | 23.101 | 28887.35 |
| **LSTM** | **All Sensors** | **28.924** | **19.960** | **21432.02** |
| LSTM | Filtered | 30.466 | 22.082 | 16901.39 |
| CNN+BiLSTM | All Sensors | 29.696 | 21.228 | 22501.43 |
| CNN+BiLSTM | Filtered | 31.803 | 22.779 | 30125.20 |

> PHM Score is an asymmetric metric — lower is better. It penalises late predictions more heavily than early predictions. Values are not normalised and are used for relative comparison only.

### Key Findings
- **LSTM with all sensors** is the best model on both FD001 (RMSE=16.125) and FD002 (RMSE=28.924)
- Sensor removal degraded performance on FD002, indicating low-variance sensors still encode operating regime information in multi-condition settings
- FD002 models show ~75% higher RMSE than FD001, directly reflecting the added complexity of 6 operating conditions

---

## Requirements

```
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
scipy>=1.7.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Download the full C-MAPSS dataset from [NASA](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and place files in `Data/`

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Open and run the notebook
```bash
jupyter notebook Code/DL_end_sem_Project.ipynb
```

> The notebook is designed to run on **Google Colab**. Update `DATA_PATH` and `BASE_PATH` variables in the notebook to point to your Google Drive paths before running.

---

## Project Report

The full project report including problem statement, methodology, ablation study results, and maintenance decision framework is available in:
```
Results/Deep_learning_project_results.pdf
```

---

## Domain

`Predictive Maintenance` · `Deep Learning` · `Time Series` · `Industrial IoT` · `NASA C-MAPSS`

---

## Author

*B.Tech / M.Tech Deep Learning End Semester Project*
