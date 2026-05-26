# Mixture-of-Experts Framework for Mid-Price Movement Prediction (Improved)

**Dataset:** FI-2010 Benchmark LOB Dataset  
**Task:** 3-class mid-price movement prediction (Up / Stationary / Down) with trading strategy, backtracking, and backtesting

---

## Quick Start

```bash
pip install -r requirements.txt
python run_experiment.py
```

---

## Project Structure

```
project/
├── data/
│   ├── load_fi2010.py       # Data loader (reads .txt, applies preprocessing)
│   ├── cross_val.py         # Anchored forward CV fold iterator
│   └── BenchmarkDatasets/   # ← Place FI-2010 dataset here
│
├── experts/
│   ├── logistic_regression.py  # LR expert (calibrated)
│   ├── xgboost_expert.py       # XGBoost expert (GPU, early stopping, calibrated)
│   └── mlp_expert.py           # MLP expert (LR scheduler, gradient clipping)
│
├── moe/
│   ├── gating_network.py    # Gating network (dropout, early stopping, meta-features)
│   └── mixture.py           # MoE forward pass
│
├── strategy/
│   ├── strategy_layer.py    # Signal strategy (EMA smoothing, tuned thresholds)
│   └── backtracking.py      # Online weight/threshold adaptation
│
├── backtest/
│   ├── engine.py            # Portfolio simulation
│   └── metrics.py           # Financial performance metrics
│
├── train.py                 # Stages 1–3 per fold
├── evaluate.py              # Stage 4: inference + backtesting
├── run_experiment.py        # Full pipeline (all 9 folds)
├── requirements.txt
└── README.md
```

---

## Dataset Setup

Place the FI-2010 dataset at `data/BenchmarkDatasets/` with this structure:

```
data/BenchmarkDatasets/
├── NoAuction/
│   ├── 1.NoAuction_Zscore/
│   │   ├── NoAuction_Zscore_Training/
│   │   │   └── Train_Dst_NoAuction_ZScore_CF_{1..9}.txt
│   │   └── NoAuction_Zscore_Testing/
│   │       └── Test_Dst_NoAuction_ZScore_CF_{1..9}.txt
│   └── 3.NoAuction_DecPre/   (optional, for real mid-price returns)
└── ...
```

Configure path at the top of `run_experiment.py`:

```python
DATASET_ROOT = "data/BenchmarkDatasets"
```

---

## Reproducibility

All random seeds are fixed via `SEED = 42` in `run_experiment.py`.
