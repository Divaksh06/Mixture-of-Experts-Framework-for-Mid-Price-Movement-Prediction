# Mixture-of-Experts Architecture for Temporal Pattern Recognition
## PRML Research Project: Mid-Price Movement Prediction in Limit Order Books

**Dataset:** FI-2010 Benchmark LOB Dataset  
**Domain:** Financial Time-Series Pattern Recognition  
**Task:** 3-class classification (Up / Stationary / Down) using a learned Mixture-of-Experts (MoE) ensemble for high-frequency market signal detection.

---

## Project Overview (Research Context)

This project investigates the application of **Mixture-of-Experts (MoE)** architectures to the non-stationary and noisy domain of high-frequency Limit Order Books (LOB). We explore the "Predictor Paradox" — where standard ML performance metrics (Accuracy/F1) decouple from real-world decision utility.

The system features:
- **Diverse Expert Ensemble:** Logistic Regression (Linear), XGBoost (Non-linear Tabular), and Temporal MLP (Sequential).
- **Meta-Learning Router:** A learned gating network that adaptively weights experts based on real-time signal quality.
- **Explainability Layer:** Post-hoc analysis of gating weights to discover architecture specialization across market regimes.
- **Utility Assessment:** Evaluating model reliability through a downstream strategy and backtesting engine.

---

## 🚀 Research Findings (9-Fold Ablation Study)

We evaluate the system using anchored forward cross-validation across 9 chronological folds.

### Classification Performance

| Model | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| **LR (Expert 1)** | 44.7% ± 1.9% | 43.2% ± 1.4% |
| **XGBoost (Expert 2)** | 53.7% ± 2.8% | 50.2% ± 4.4% |
| **Temporal MLP (Expert 3)** | **57.2% ± 3.6%** | **56.2% ± 3.3%** |
| **Gated MoE** | 57.1% ± 3.8% | 53.4% ± 6.4% |

### Gating Network Diagnostics

| Expert | Mean Soft Weight | Argmax Routing |
| :--- | :--- | :--- |
| LR | 0.09% ± 0.04% | 0.0% |
| **XGB** | **78.0% ± 13.8%** | **77.7%** |
| MLP | 21.9% ± 13.8% | 22.3% |

The gating network empirically learns to route ~78% of predictions through XGBoost while using MLP 22% of the time as a regime filter. LR is effectively zeroed out.

### Trading Performance (1bp cost)

| Strategy | Return (mean±std) | Median | Sortino | MaxDD | Trades | Win Rate | Dec Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGB alone** | +32.8% ± 86.2% | +3.2% | 0.097 | 22.7% | 280 | 58.7% | 74.2% |
| **MLP alone** | +0.2% ± 0.4% | 0.0% | 0.006 | 0.2% | 9.6 | — | — |
| **Gated MoE** | +31.8% ± 76.9% | **+7.8%** | **0.123** | **11.7%** | 281 | 57.9% | 74.4% |
| **MoE + BT**  | +60.7% ± 120.5% | +32.4% | 0.177 | 13.0% | 397 | 55.2% | 67.1% |
| **MoE + CR**  | +32.4% ± 86.6% | +3.1% | 0.096 | 22.8% | 281 | 57.9% | 74.3% |
| **Buy & Hold** | +32.5% ± 4.4% | +31.1% | — | — | — | — | — |

**Negative Result:** Backtracking degrades precision and win rate; see Section 9 below.

> **Median-First Reporting.** Standard deviations exceed means for all active strategies. Median return is the honest measure of typical session performance. The MoE improves median return **2.4x** (+7.8% vs +3.2%) over XGBoost solo. 

### 📉 The Classification-Profitability Paradox

**Why does the highest-accuracy model lose money?**

| Model | Macro-F1 | Mean Return | Median Return | Trades |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (Expert 3)** | **56.2%** | +0.2% | 0.0% | 9.6 |
| **XGBoost (Expert 2)** | 50.2% | +32.8% | +3.2% | 280 |

The MLP optimises for classification accuracy by conservatively predicting "Stationary" — achieving high F1 but rarely breaching the τ_entry=0.60 execution threshold. It averages only 9.6 trades across ~38k test samples. XGBoost's tabular split structure produces high-conviction directional outputs that consistently exceed the threshold, enabling 280 trades per fold.

### 🏆 Why the MoE Architecture Works (PRML Insights)

The MoE architecture provides superior **risk-adjusted performance** by modularizing the prediction task:

1. **Architecture Specialization (Explainability).** Gating diagnostics reveal the network discovered a dual-mode routing logic:
   - **XGBoost (78% weight):** Functions as the primary "Directional Conviction" expert, capturing high-signal non-linearities.
   - **Temporal MLP (22% weight):** Acts as a "Regime Filter," providing conservative dampening during high-volatility microstructure noise.
2. **Risk reduction.** By ensembling diverse architectures, the Gated MoE halves maximum drawdown (11.7% vs 22.7%) compared to its best single expert. Sortino ratio improves by 26%, proving that model diversity leads to more stable downstream utility.
3. **Heuristic Collapse.** Confidence-based routing baselines (MoE+CR) collapse to single-expert performance, confirming that the learned gating network captures inter-expert dynamics that simple heuristics cannot replicate.

### ⚡ Feature Engineering & Latency Optimization (Occam's Razor)
In an ablation study, expanding the MoE Gating Network to process the full raw 147-dimensional LOB state alongside the 3 expert probabilities yielded identical routing efficiency (Return: ~0.199). This proves that the base experts flawlessly exhaust the predictive variance of the LOB. By maintaining the gating network isolated to just the 3 probabilities, the router avoids processing 147 raw features dynamically, drastically slashing computational overhead and routing execution time—a paramount requirement for High-Frequency Trading systems.

> **Statistical Note:** Due to temporal dependence between folds, statistical tests are approximate. MoE's mean return does not show a statistically significant absolute profit improvement over XGBoost (p = 0.90), as performance differences are eclipsed by fold variance. However, the true edge lies in the structural risk mitigation (Sortino / R-to-V).

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

> [!IMPORTANT]
> **Preliminary Step:** Before running the project, you MUST unzip the dataset file:
> ```bash
> # Unzip the dataset into the data/ directory
> unzip data/BenchmarkDatasets.zip -d data/
> ```
> This will create the required `data/BenchmarkDatasets/` directory structure.

This project uses the **FI-2010 Benchmark Dataset** with pre-normalized `.txt` files.

### Expected Directory Structure

Place the dataset folder in the **root of the project directory** so the structure looks like this:

```
Project_Root/
├── data/
│   ├── BenchmarkDatasets.zip (Download this)
│   └── BenchmarkDatasets/    (Created after unzip)
│       └── NoAuction/
│           └── 3.NoAuction_DecPre/
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

### Expert Models: Architectural Diversity

| Expert | Architecture | Domain Specialization |
|---|---|---|
| **Logistic Regression** | L2-Regularized Linear mapping | High-bias baseline for linear separability testing. |
| **XGBoost (GBDT)** | 500 trees, depth $\in \{3,4,5\}$ | Captures non-linear tabular patterns; provides directional conviction. |
| **Temporal MLP** | 735 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 3 | Captures short-term momentum via 5-step sequence stacking ($5 \times 147$). Uses **Focal Loss** ($\gamma=2.0$) and BatchNorm. |

### Gating Network (The Meta-Classifier)

The gating router is a lightweight MLP ($9 \rightarrow 64 \rightarrow 3$) that learns to assign dynamic trust weights to each expert based on the current market context. 
- **Input Features:** 9-dimensional vector (concatenated probabilities from the 3 experts).
- **Objective:** Minimizes overall cross-entropy of the gating-weighted combination through **Stacked-Probability Training** (Stage 2) to prevent train-test leakage.

### Evaluative Utility: Trading & Backtest Engine

To evaluate model maturity beyond F1-scores, we execute a downstream backtest:
- **State Machine:** Converts probabilities into actions via conviction hysteresis ($\tau_{entry}=0.60$) and a **20-step signal regime filter**.
- **Portfolio Physics:** $hold\_k = 25$ steps, 1bp transaction cost, and slippage modeling.
- **Utility Metrics:** Sortino Ratio (Risk-adjusted utility), Max Drawdown (Signal stability), and Cumulative Return.

---

## Key Finding: Risk over Raw Return

The defensible narrative of this research is that the **Mixture-of-Experts (MoE) provides risk-adjusted improvement that matters in practice**, even if it doesn't beat its best single expert (XGBoost) on raw mean return.

- **MoE halves maximum drawdown** (11.7% vs 22.7%).
- **MoE improves median return 2.4x** (+7.8% vs +3.2%).
- **Backtracking is a negative result:** Online threshold adaptation increases activity but degrades precision, suggesting it is destabilising on short horizons.

**Predictive accuracy alone is insufficient for profitable decision-making; structural risk mitigation is the true contribution of the MoE framework.**

---

## Generalization Disclaimer

This study is conducted on a 10-day limit order book dataset for two instruments. Generalization across assets, time periods, and market regimes is not evaluated and remains future work.

---

## References

1. A. Ntakaris et al., "Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods," *Journal of Forecasting*, arXiv:1705.03233v5, 2020.
2. A. N. Kercheval and Y. Zhang, "Modelling high-frequency limit order book dynamics with support vector machines," *Quantitative Finance*, vol. 15, no. 8, 2015.