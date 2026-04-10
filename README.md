# Mixture of Experts Framework for Mid-Price Movement Prediction

This repository implements an advanced **Mixture-of-Experts (MoE)** machine learning pipeline designed to predict short-term mid-price movements within the FI-2010 High-Frequency Trading (HFT) Limit Order Book (LOB) dataset.

Unlike traditional classifiers that maximize theoretical accuracy formulas (which helplessly favor "Stationary" predictions without ever extracting capital), this project routes localized models through an automated gating network and a mathematical algorithmic trading state machine to translate probabilities into compounding profit.

## 🧠 Model Architecture

The framework decouples statistical capability from financial execution constraints using a localized structure:

### 1. The Expert Models
- **Expert 1: Logistic Regression:** A baseline linear architecture designed to exclusively bind against mathematically engineered momentum ratios.
- **Expert 2: XGBoost:** Gradient-boosted sequence trees utilizing shallow leaf boundaries to dynamically capture spatial tabular interactions without overfitting microstructure noise.
- **Expert 3: Temporal PyTorch MLP:** A dense multi-layer perceptron utilizing localized chronological `lookback=5` feature mapping across 735 dimensions. It explicitly trains on an asymmetrical `FocalLoss` derivative to penalize stationary guessing.

### 2. MoE Gating Network
A centralized probability matrix utilizing a Softmax router to dictate trust scales dynamically across the three experts depending on the specific state of the order book.

### 3. Financial Strategy State Machine
A latency-simulating trading engine translating algorithmic predictions into physical executed positions:
- **Hysteresis Band:** Uses explicit $\tau_{entry}=0.60$ and $\tau_{exit}=0.15$ constraints to filter weak-probability executions.
- **Regime Filter:** A structural 20-step rolling tracking queue (`s_avg > 0.55`) that mathematically locks the engine from triggering inside stochastic "sideways" market traps, eliminating hundreds of losing noise-based trades.

---

## 🚀 The Predictor Paradox Results (9-Fold Ablation)

Our Anchored Time-Series Ablation physically isolates the gap between mathematical accuracy and actual financial extraction:

| Model | Classification Accuracy | Return |
| :--- | :--- | :--- |
| **B&H Benchmark** | -- | +32.5% |
| **Temporal MLP** | **57.1%** | -5.8% |
| **Best Single (XGB)** | 53.6% | +43.6% |
| **Overall MoE** | 57.8% | **+46.5%** |

*Note: The Neural Network achieves extreme accuracy by guessing "Stationary," missing all physical directional trades (losing capital). The MoE synergizes XGBoost's directional depth splits with the MLP's baseline boundaries to extract maximum structural profit globally.*

---

## Package Requirements

All required libraries are listed in `requirements.txt`. The key dependencies are:

| Package | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.23 | Array operations, numerical computing |
| `pandas` | >= 1.5 | Data loading and manipulation |
| `scikit-learn` | >= 1.2 | Logistic Regression, preprocessing, metrics |
| `xgboost` | >= 1.7 | Gradient-boosted tree expert |
| `torch` | >= 2.0 | MLP expert and gating network (PyTorch) |
| `scipy` | >= 1.10 | Statistical utilities |
| `matplotlib` | >= 3.6 | Visualisation |
| `seaborn` | >= 0.12 | Enhanced plotting |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** If you are using a GPU, install the appropriate CUDA-compatible version of PyTorch before running `pip install -r requirements.txt`. (The PyTorch script detects and activates NVIDIA CUDA automatically).

---

## Dataset Setup

This project uses the **FI-2010 Benchmark Dataset**.

### Expected Directory Structure
Crucially, this project explicitly evaluates the raw unnormalized `NoAuction_DecPre` dataset in order to precisely calculate non-linear spatial formulas (Spread, L1 & L5 Order Imbalance).
Place the dataset folder in the **root of the project directory** so the structure looks like this:

```
├── data/
│   ├── BenchmarkDatasets/
│   │   └── NoAuction/
│   │       ├── 1.NoAuction_Zscore/
│   │       ├── 2.NoAuction_MinMax/
│   │       └── 3.NoAuction_DecPre/
│   │           ├── NoAuction_DecPre_Training/
│   │           │   └── Train_Dst_NoAuction_DecPre_CF_1.txt ... CF_9.txt
│   │           └── NoAuction_DecPre_Testing/
│   │               └── Test_Dst_NoAuction_DecPre_CF_1.txt ... CF_9.txt
├── experts/
├── moe/
├── strategy/
├── backtest/
├── train.py
├── evaluate.py
├── run_experiment.py
├── requirements.txt
└── README.md
```

### Dataset File Format
- **Columns 0–143**: LOB feature arrays (augmented to 147 internally via Imbalance and Spread)
- **Column 148**: Label for prediction horizon k=10
- **Labels:** 0 = Up, 1 = Stationary, 2 = Down

---

## Project Structure

```text
├── data/
│   ├── load_fi2010.py       # Data loader: extracts and calculates Imbalance and Spread
│   └── cross_val.py         # Anchored forward CV fold iterator (folds 1–9)
│
├── experts/
│   ├── logistic_regression.py  # LR expert: L2 constraint, linear mapping
│   ├── xgboost_expert.py       # XGBoost expert: Tree boosting (GPU-aware)
│   └── mlp_expert.py           # Temporal MLP (PyTorch): lookback=5 sequence stack, Focal Loss
│
├── moe/
│   ├── gating_network.py    # Gating network (PyTorch): dynamic Softmax trust evaluation
│   └── mixture.py           # Weighted MoE probability interpolation
│
├── strategy/
│   ├── strategy_layer.py    # Decision rules: Evaluates Conviction thresholds + Regime Filter queue
│   └── backtracking.py      # Error tracking and adjustment (Phase-1 legacy)
│
├── backtest/
│   ├── engine.py            # Financial latency simulator (hold_k=25, slippage logic)
│   └── metrics.py           # Evaluator: Sharpe, Maximum Drawdown, Hit Ratio
│
├── run_experiment.py        # Central master script driving full ablation across 9 CV Folds
│
├── fi2010_eda.ipynb         # Comprehensive physical Exploratory Data Analysis & Feature distribution
├── report.tex               # Formal IEEE Latex academic submission 
└── presentation.md          # Outline of defense slides
```

---

## Reproducibility

All global randomness is strictly anchored via seed locking directly inside `run_experiment.py`:

```python
SEED = 42
```
This forces `numpy`, `torch`, `random`, and `xgboost` into entirely deterministic boundaries to guarantee identically verifiable pipeline replication across any compliant execution node.

---

## References

1. A. Ntakaris et al., "Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods," *Journal of Forecasting*, arXiv:1705.03233v5, 2020.
2. A. N. Kercheval and Y. Zhang, "Modelling high-frequency limit order book dynamics with support vector machines," *Quantitative Finance*, vol. 15, no. 8, 2015.