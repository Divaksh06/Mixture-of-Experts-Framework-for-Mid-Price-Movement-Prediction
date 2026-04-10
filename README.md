# Mixture-of-Experts Framework for Mid-Price Movement Prediction
 
**Dataset:** FI-2010 Benchmark LOB Dataset  
**Task:** 3-class mid-price movement prediction (Up / Stationary / Down) with trading strategy generation, backtracking, and backtesting

---

## Project Description

This project implements a **Mixture-of-Experts (MoE)** framework for predicting short-term mid-price movements from high-frequency limit order book (LOB) data. Three expert classifiers — Logistic Regression, XGBoost, and a Multi-Layer Perceptron — are combined via a learned gating network that adaptively weights each expert based on the input market state.

The system goes beyond pure classification and includes:
- A **strategy layer** that converts probabilistic predictions into exact physical trades using hysteresis bands ($\tau_{entry}=0.60$) and a **20-step signal regime filter** to block sideways market noise.
- A **backtracking module** that online-adjusts expert weights and strategy thresholds based on observed prediction errors.
- A **backtesting engine** that simulates a trading portfolio ($hold_k=25$ duration constraint) and evaluates financial performance (return-to-volatility ratio, Sortino ratio, cumulative return, max drawdown, win rate).

---

## 🚀 The Predictor Paradox Results (9-Fold Ablation)

We report results across 9 chronological folds. All reported ratios are computed on tick-level returns. Annualized values are approximate and depend on frequency assumptions.

| Model | Accuracy | Macro-F1 | Return | R-to-V Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Directional Benchmark (Long-Only)** | -- | -- | +32.5% | -- | -- |
| **LR (Expert 1)** | 48.2% | 38.4% | -- | -- | -- |
| **Temporal MLP** | 57.1% | **56.2%** | -5.8% | -0.12 | 6.0% |
| **Best Single (XGB)** | 53.6% | 49.9% | +43.6% | 1.45 | 19.2% |
| **Overall MoE** | **57.8%** | 54.1% | **+46.5%** | **1.62** | **22.2%** |

> **Note:** Buy-and-Hold is not directly comparable due to continuous exposure and absence of transaction costs. We do not claim outperformance vs B&H due to differing exposure profiles.

### 📉 The MLP "Accuracy vs. Profit" Paradox Explained
**Why did the PyTorch Neural Network get the highest accuracy (`57.1%`) but lose money (`-5.8%`)?**
In Limit Order Books (like FI-2010), the overwhelming majority of price movements are "Stationary". A neural network optimizing mathematically for raw correctness realizes that safely guessing "Stationary" on almost every tick guarantees high theoretical accuracy. However, our algorithmic trading engine requires a physical probability threshold of $\tau_{entry}=0.60$ to actually deploy capital. 
Because the MLP acts conservatively, it rarely breached that threshold. It only attempted 10 actual directional trades across the entire dataset, almost all of which were caught in microstructure latency traps, bleeding `-5.8%` to transaction costs and spread-crossing.

### 🏆 Why the MoE Wins 
XGBoost, by contrast, is aggressively mapping tabular spatial splits on our fractional `DecPre` engineered features. It bypassed threshold limits easily, executing **279** aggressive trades. 
By utilizing the **Gating Network**, the overall MoE ensemble learned to mathematically synergize them: 
- It used **XGBoost** to execute heavy-conviction directional movements.
- It used the **MLP** to recognize baseline sideways non-stationarity, actively suppressing XGBoost from over-trading during weak volume phases.

| Engine | Total Executed Trades | Execution Win Rate | Raw Return |
| :--- | :--- | :--- | :--- |
| **Temporal MLP** | 10 | ~0.0% | -5.8% |
| **XGBoost (Solo)** | 279 | 55.2% | +43.6% |
| **Gated MoE Engine** | 280 | **56.8%** | **+46.5%** |

This evidence suggests that assembling diverse mathematical priors inside an MoE structure can perform comparably to or marginally better than the best single model, though statistical significance was not established (p = 0.38 across 9 folds).

### 📊 Transaction Cost Sensitivity

Transaction costs are varied as a stress test rather than an exact simulation of real execution conditions:

| Cost | MoE Return | XGB Return |
| :--- | :--- | :--- |
| **1 bp** | +46.5% | +43.6% |
| **3 bp** | +29.1% | +26.7% |

*Performance degrades under higher costs but relative ranking is preserved.*

### 🔬 Gating Ablation

| Model | Return |
| :--- | :--- |
| XGB Solo | +43.6% |
| Equal Ensemble (1/3 each) | +0.03% |
| Weighted Ensemble (F1-based) | +0.12% |
| **Gated MoE (Ours)** | **+46.5%** |

*MoE performs comparably to the best single model (XGBoost). Gated MoE vs Weighted Ensemble suggests gating shows potential benefit.*

> **Statistical Note:** Due to temporal dependence between folds, statistical tests are approximate. MoE does not show statistically significant improvement over XGBoost (p = 0.38). Performance differences are within variance across folds.

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
NORMALIZATION = "NoAuction_DecPre"   # Default normalization variant used (Requires raw integer volumes!)
```

---

## Project Structure

```
Group_XX/
├── data/
│   ├── load_fi2010.py       # Data loader: explicitly calculates L1 & L5 Imbalance and Spread
│   └── cross_val.py         # Anchored forward CV fold iterator (folds 1–9)
│
├── experts/
│   ├── logistic_regression.py  # LR expert: L2 constraint, linear mapping
│   ├── xgboost_expert.py       # XGBoost expert: Tree boosting (GPU-aware)
│   └── mlp_expert.py           # Temporal MLP (PyTorch): lookback=5 sequence stack (dim=735), Focal Loss
│
├── moe/
│   ├── gating_network.py    # Gating network (PyTorch): dynamic Softmax trust evaluation
│   └── mixture.py           # Weighted MoE probability interpolation
│
├── strategy/
│   ├── strategy_layer.py    # Decision rules: Evaluates Conviction thresholds + 20-step Regime Filter queue
│   └── backtracking.py      # Error tracking and adjustment 
│
├── backtest/
│   ├── engine.py            # Portfolio simulation (hold_k=25, slippage logic)
│   └── metrics.py           # Evaluator: Return-to-Volatility, Sortino, Max Drawdown, Hit Ratio
│
├── train.py                 # Stages 1–3: expert training + gating network training per fold
├── evaluate.py              # Stage 4: inference + strategy + backtracking + backtesting
├── run_experiment.py        # Central master script driving full ablation across 9 CV Folds
├── requirements.txt
├── fi2010_eda.ipynb         # Comprehensive physical Exploratory Data Analysis
├── presentation.md          # Deck Outline
├── report.tex               # LaTeX Academic Report
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
- **Financial metrics** (mean ± std): Cumulative Return, Return-to-Volatility Ratio, Sortino Ratio, Max Drawdown, Win Rate
- **Benchmarks**: Directional Benchmark (Long-Only), Passive Signal Benchmark
- **Cost Sensitivity**: 1bp vs 3bp transaction cost comparison
- **Gating Ablation**: Equal ensemble, F1-weighted ensemble, vs Gated MoE
- **Statistical Rigor**: Paired t-test (MoE vs XGB), win count, Spearman correlation (F1 vs Return)

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
| XGBoost | Gradient-boosted trees | n_estimators=500, max_depth ∈ {3,4,5}, GPU-Hist |
| MLP (PyTorch) | Temporal Neural network | 735→256→128→3, Dropout=0.3, Focal Loss |

### Gating Network

A lightweight neural network (144→64→3, softmax output) learns to assign input-dependent weights (w1, w2, w3) to the three experts. It is trained on 9-dimensional stacked expert probability outputs.

### Strategy Layer

Converts the final probability vector into a trading action using:
- **Conviction Hysteresis**: Rejects weak trades by enforcing $\tau_{entry}=0.60$ and $\tau_{exit}=0.15$.
- **Signal Regime Filter**: A 20-step rolling queue ($\bar{s}_t > 0.55$) inherently blocking stochastic sideways market chop.

### Backtracking Module

Maintains a rolling buffer of 100 recent predictions and outcomes. Every 50 samples, it:
- Reweights experts proportional to their recent accuracy
- Adjusts strategy thresholds based on false positive / missed opportunity rates (clipped to [0.45, 0.85])

### Backtesting Engine

Simulates a long-only trading account with:
- Initial capital: 10,000 units
- Minimum holding period: $hold\_k = 25$ steps to mirror spread-crossing latency constraints
- Transaction cost: 1 basis point per trade side (3bp sensitivity also evaluated)
- Metrics: Cumulative Return, Return-to-Volatility Ratio, Sortino Ratio, Max Drawdown, Win Rate, Decision Accuracy
- **Metric Interpretation:** All reported ratios are computed on tick-level returns and are non-annualized. Scaled Sharpe values assume 1-second tick frequency (annual factor = $\sqrt{252 \times 6.5 \times 3600}$) and should not be compared to industry Sharpe ratios due to unrealistic scaling assumptions.

---

## Key Finding

The Mixture-of-Experts model improves classification performance but does not consistently translate this into superior trading returns.

Across 9 chronological folds:
- MoE achieves higher mean return than XGBoost
- However, median return is lower
- No statistically significant improvement is observed (p = 0.38)

This highlights a central challenge in financial machine learning:

**Predictive accuracy alone is insufficient for profitable decision-making under transaction costs and execution constraints.**

---

## Generalization Disclaimer

This study is conducted on a 10-day limit order book dataset for two instruments. Generalization across assets, time periods, and market regimes is not evaluated and remains future work.

---

## References

1. A. Ntakaris et al., "Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods," *Journal of Forecasting*, arXiv:1705.03233v5, 2020.
2. A. N. Kercheval and Y. Zhang, "Modelling high-frequency limit order book dynamics with support vector machines," *Quantitative Finance*, vol. 15, no. 8, 2015.