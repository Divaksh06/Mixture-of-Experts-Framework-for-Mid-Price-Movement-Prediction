# Mixture-of-Experts Framework for Mid-Price Movement Prediction
 
**Dataset:** FI-2010 Benchmark LOB Dataset  
**Task:** 3-class mid-price movement prediction (Up / Stationary / Down) with trading strategy generation, backtracking, and backtesting

---

## Project Description

This project implements a **Mixture-of-Experts (MoE)** framework for predicting short-term mid-price movements from high-frequency limit order book (LOB) data. Three expert classifiers — Logistic Regression, XGBoost, and a Multi-Layer Perceptron — are combined via a learned gating network that adaptively weights each expert based on the input market state.

The system goes beyond pure classification and includes:
- A **strategy layer** that converts probabilistic predictions into Buy / Hold / Sell signals using confidence filtering
- A **backtracking module** that online-adjusts expert weights and strategy thresholds based on observed prediction errors
- A **backtesting engine** that simulates a trading portfolio and evaluates financial performance (Sharpe ratio, cumulative return, max drawdown, win rate)

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

> **Note:** If you are using a GPU, install the appropriate CUDA-compatible version of PyTorch from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## Dataset Setup

This project uses the **FI-2010 Benchmark Dataset** with pre-normalized `.txt` files.

### Expected Directory Structure

Place the dataset folder in the **root of the project directory** so the structure looks like this:

```
Group_XX/
├── BenchmarkDatasets/
│   ├── Auction/
│   │   ├── 1.Auction_Zscore/
│   │   │   ├── Auction_Zscore_Training/
│   │   │   │   ├── Train_Dst_Auction_Zscore_CF_1.txt
│   │   │   │   └── ... (CF_1 through CF_9)
│   │   │   └── Auction_Zscore_Testing/
│   │   │       ├── Test_Dst_Auction_Zscore_CF_1.txt
│   │   │       └── ... (CF_1 through CF_9)
│   │   ├── 2.Auction_MinMax/
│   │   └── 3.Auction_DecPre/
│   └── NoAuction/
│       ├── 1.NoAuction_Zscore/
│       │   ├── NoAuction_Zscore_Training/
│       │   │   └── Train_Dst_NoAuction_Zscore_CF_1.txt ... CF_9.txt
│       │   └── NoAuction_Zscore_Testing/
│       │       └── Test_Dst_NoAuction_Zscore_CF_1.txt ... CF_9.txt
│       ├── 2.NoAuction_MinMax/
│       └── 3.NoAuction_DecPre/
├── data/
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

Each `.txt` file contains space/tab-separated values where:
- **Columns 0–143** (144 total): Feature values (already normalized)
- **Column 148** (0-indexed): Label for prediction horizon k=10 — used in this project
- **Labels:** 1 = Up, 2 = Stationary, 3 = Down

### Configuring the Dataset Path

The dataset root path is configurable at the top of `run_experiment.py`:

```python
DATASET_ROOT = "BenchmarkDatasets"   # Change this if your dataset is elsewhere
NORMALIZATION = "NoAuction_Zscore"   # Default normalization variant used
```

---

## Project Structure

```
Group_XX/
├── data/
│   ├── load_fi2010.py       # Data loader: reads .txt files, applies preprocessing
│   └── cross_val.py         # Anchored forward CV fold iterator (folds 1–9)
│
├── experts/
│   ├── logistic_regression.py  # LR expert: fit, predict_proba, evaluate
│   ├── xgboost_expert.py       # XGBoost expert: fit, predict_proba, evaluate
│   └── mlp_expert.py           # MLP expert (PyTorch): model, train loop, predict_proba
│
├── moe/
│   ├── gating_network.py    # Gating network (PyTorch): maps 9-dim stacked probs → weights
│   └── mixture.py           # MoE forward pass: P_final = sum(wi * Pi)
│
├── strategy/
│   ├── strategy_layer.py    # Decision rules: Buy / Hold / Sell with confidence margin filter
│   └── backtracking.py      # Memory buffer, error tracking, weight and threshold updates
│
├── backtest/
│   ├── engine.py            # Portfolio simulation (step-by-step)
│   └── metrics.py           # Cumulative return, Sharpe ratio, MDD, Win Rate, Decision Accuracy
│
├── train.py                 # Stages 1–3: expert training + gating network training per fold
├── evaluate.py              # Stage 4: inference + strategy + backtracking + backtesting
├── run_experiment.py        # Full pipeline across all 9 CV folds with aggregated results
├── requirements.txt
└── README.md
```

---

## Run Instructions

### Step 1 — Set Up Environment

It is recommended to use a virtual environment:

```bash
# Create virtual environment (optional but recommended)
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```


### Step 3 — Run the Full Experiment

```bash
python run_experiment.py
```

This single command executes the complete pipeline across all 9 cross-validation folds:
1. Loads and preprocesses data for each fold
2. Trains the three expert models (LR, XGBoost, MLP)
3. Trains the gating network on stacked expert probabilities
4. Runs inference with the strategy layer, backtracking module, and backtesting engine
5. Aggregates and prints results

### Step 4 — View Results

After completion, the terminal will display a results summary including:

- **Classification metrics** (mean ± std across 9 folds): Accuracy, Macro-F1, per-class F1
- **Financial metrics** (mean ± std): Cumulative Return, Sharpe Ratio, Max Drawdown, Win Rate, Decision Accuracy
- **Comparison tables**: Individual experts vs. MoE ensemble, MoE with vs. without backtracking, Strategy vs. buy-and-hold benchmark

Model checkpoints for each fold are saved to `results/fold_{t}/`.

---

## Reproducibility

All random seeds are fixed at the start of `run_experiment.py`:

```python
RANDOM_SEED = 42
```

This seeds `numpy`, `torch`, `random`, and `xgboost` to ensure fully reproducible results across runs.

---

## Method Summary

### Expert Models

| Expert | Type | Key Hyperparameters |
|---|---|---|
| Logistic Regression | Linear classifier | C ∈ {0.01, 0.1, 1, 10}, lbfgs solver |
| XGBoost | Gradient-boosted trees | n_estimators=200, max_depth ∈ {3,5,7} |
| MLP (PyTorch) | Neural network | 144→256→128→3, Dropout=0.3, Adam |

### Gating Network

A lightweight neural network (144→64→3, softmax output) learns to assign input-dependent weights (w1, w2, w3) to the three experts. It is trained on 9-dimensional stacked expert probability outputs.

### Strategy Layer

Converts the final probability vector into a trading action using:
- A **confidence margin filter** (margin between top-2 class probs > δ = 0.10)
- **Class-specific thresholds**: θ_buy = 0.55, θ_sell = 0.55

### Backtracking Module

Maintains a rolling buffer of 100 recent predictions and outcomes. Every 50 samples, it:
- Reweights experts proportional to their recent accuracy
- Adjusts strategy thresholds based on false positive / missed opportunity rates (clipped to [0.45, 0.85])

### Backtesting Engine

Simulates a long-only trading account with:
- Initial capital: 10,000 units
- Transaction cost: 1 basis point per trade side
- Metrics: Cumulative Return, Sharpe Ratio, Max Drawdown, Win Rate, Decision Accuracy

---

## References

1. A. Ntakaris et al., "Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods," *Journal of Forecasting*, arXiv:1705.03233v5, 2020.
2. A. N. Kercheval and Y. Zhang, "Modelling high-frequency limit order book dynamics with support vector machines," *Quantitative Finance*, vol. 15, no. 8, 2015.