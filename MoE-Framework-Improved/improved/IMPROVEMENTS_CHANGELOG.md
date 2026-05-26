# Improvements Changelog — Previous → Improved Codebase

This document lists every change made from the original codebase to the improved version, grouped by file/module.

---

## 1. `data/load_fi2010.py` & `data/cross_val.py` — **NEW FILES (Reconstructed)**

**Problem:** These files were missing from the original zip. The pipeline imports `from data.load_fi2010 import load_fold` and `from data.cross_val import anchored_forward_cv` but no `data/` directory existed.

**Fix:** Reconstructed both files based on:
- The `load_fold` function present in the EDA notebook (`EDA_FI2010.ipynb` Cell 4)
- The README's documented data layout
- The import signature in `train.py` (which expects 7 return values including k=5 labels)

**Changes:**
- `load_fi2010.py`: Complete data loader with k=10 *and* k=5 horizon labels
- `load_mid_prices()`: Attempts DecPre variant first for realistic prices, falls back to Zscore proxy
- `cross_val.py`: Simple iterator yielding fold indices 1…9

---

## 2. `experts/logistic_regression.py` — Probability Calibration

| Aspect | Original | Improved |
|--------|----------|----------|
| C search grid | `[0.01, 0.1, 1.0, 10.0]` | `[0.001, 0.01, 0.1, 1.0, 10.0]` — wider search |
| CV scoring | `n_jobs=None` | `n_jobs=-1` — parallel scoring |
| Calibration | None | **Isotonic regression** via `CalibratedClassifierCV` — better-calibrated probabilities improve gating network and strategy layer |

**Why:** Raw LR probabilities are often overconfident. Isotonic calibration maps predicted probabilities to observed frequencies, so a prediction of P(Up)=0.6 actually corresponds to ~60% Up events in practice.

---

## 3. `experts/xgboost_expert.py` — Early Stopping + GPU + Calibration

| Aspect | Original | Improved |
|--------|----------|----------|
| n_estimators | 500 (no early stopping) | **800 with early stopping** — trains until convergence, not a fixed count |
| Tree method | `auto` (CPU) | **`gpu_hist`** when CUDA detected, `hist` otherwise — utilises 10 GB VRAM |
| Regularisation | Default | Added `reg_alpha=0.1`, `min_child_weight=3` — prevents overfitting |
| Depth search | `[3, 4, 5]` | `[3, 4, 5, 6]` — wider search |
| Final training | Trained on full data without validation | **10% held out for early stopping** in final training |
| Calibration | None | **Isotonic calibration** post-training |

**Why:** The original XGBoost always trained all 500 trees even if validation loss stopped improving. Early stopping prevents overfitting and saves compute. GPU acceleration significantly speeds up training (XGBoost's `gpu_hist` works well within 10 GB VRAM for this dataset size).

---

## 4. `experts/mlp_expert.py` — LR Scheduler + Gradient Clipping + Verbose Control

| Aspect | Original | Improved |
|--------|----------|----------|
| Focal Loss γ | 3.0 | **2.0** — less aggressive down-weighting |
| LR Scheduler | None | **ReduceLROnPlateau** (factor=0.5, patience=3) — adapts learning rate when validation F1 plateaus |
| Gradient clipping | None | **clip_grad_norm = 1.0** — prevents gradient explosions |
| Weight decay | None | **1e-5** — mild L2 regularisation |
| Early stopping patience | 5 epochs | **7 epochs** — more room to recover |
| Max epochs | 50 | **60** — combined with LR scheduling, allows better convergence |
| Temperature scaling | None | **Learnable temperature parameter** for calibrated probability output |
| Verbose flag | Always prints all epochs | **`verbose=False` for Stage 2 inner training** — fixes the output interleaving bug |
| pin_memory | No | **Yes** for DataLoader — faster GPU transfer |

**Critical bug fix — "Early stopping in Stage 2" output interleaving:**
The original code printed epoch progress during Stage 2 inner MLP training (9 inner MLPs). This output interleaved with Stage 3 messages, confusing the user. The new `verbose` parameter suppresses this output when running inner-fold training.

---

## 5. `moe/gating_network.py` — Meta-Features + Dropout + Early Stopping

| Aspect | Original | Improved |
|--------|----------|----------|
| Input dim | 9 (stacked probs only) | **12** = 9 stacked probs + 3 expert-agreement std features |
| Dropout | None | **0.2 dropout** after hidden layer |
| Weight decay | None | **1e-4** L2 regularisation |
| Training | Fixed 30 epochs, no validation | **Early stopping** (patience=8) on 15% validation split |
| Max epochs | 30 | **50** (with early stopping, actual epochs are typically fewer) |

**Meta-features:** For each class c ∈ {Up, Stationary, Down}, we compute the standard deviation of [P_LR(c), P_XGB(c), P_MLP(c)]. High std = experts disagree → gating should be cautious. Low std = experts agree → gating can be confident. These 3 additional features help the gating network make more informed weighting decisions.

---

## 6. `strategy/strategy_layer.py` — Tuned Thresholds + EMA Smoothing

| Aspect | Original | Improved |
|--------|----------|----------|
| `tau_entry` | 0.60 (extremely conservative) | **0.35** — allows more trades when signal is moderately strong |
| `tau_exit` | 0.15 | **0.10** — exits only when signal is very weak |
| `theta_buy` / `theta_sell` | 0.75 (too restrictive) | **0.55** — calibrated probabilities are more accurate, so lower thresholds are appropriate |
| Regime warmup | 20 ticks | **10 ticks** — faster warmup |
| Regime threshold | 0.55 | **0.45** — less restrictive regime filter |
| Signal smoothing | None | **EMA (α=0.3)** — smooths the P(up)−P(down) signal to reduce whipsaw trading |
| Threshold clamp range | [0.50, 0.90] | **[0.40, 0.80]** — wider adaptation range |

**Why the original thresholds were problematic:**
With `tau_entry=0.60`, the signal P(up)−P(down) had to exceed 0.60 — meaning P(up) needed to be ~0.80. Combined with `theta_buy=0.75`, almost no predictions passed both conditions, resulting in very few trades (turnover ~4%). The improved thresholds, combined with calibrated probabilities, allow the strategy to act on genuine moderate-confidence signals.

---

## 7. `strategy/backtracking.py` — Stabilised Adaptation

| Aspect | Original | Improved |
|--------|----------|----------|
| Buffer size (N_buf) | 100 | **200** — more data for stable accuracy estimates |
| Update frequency (N_upd) | 50 | **100** — less frequent but more reliable updates |
| theta_reverse | 0.70 | **0.65** — slightly more responsive to reversals |
| Weight clamping | None | **[0.10, 0.60]** — no expert can dominate or be zeroed out |
| FP rate computation | `buy_fp / n` | **`buy_fp / max(buy_actions, 1)`** — correct conditional rate |
| error_margin check | `abs(pred - true) <= 0.05` (useless for int labels) | **Exact match** `pred == true` — cleaner and semantically correct |

---

## 8. `backtest/engine.py` — Reduced Lockout + Lower Costs

| Aspect | Original | Improved |
|--------|----------|----------|
| hold_k (min hold) | 25 ticks | **15 ticks** — more responsive position management |
| c_spread | 0.00015 | **0.00010** — more realistic for liquid Finnish equities |
| close_position | Didn't track hold_durations | **Now tracks** hold_durations for closed positions |

---

## 9. `backtest/metrics.py` — Minor

- `turnover()` now also counts 'Flat' actions (closing is a transaction).
- No other changes; metrics were already correct.

---

## 10. `train.py` — Clean Logging + Stage 2 Fix

| Aspect | Original | Improved |
|--------|----------|----------|
| Logging | `print()` without flush | **`flush=True`** on all prints — prevents interleaved output |
| Stage 2 MLP | `MLPExpert()` (verbose=True) | **`MLPExpert(verbose=False)`** — suppresses inner-fold epoch logging |
| Stage 2 progress | No progress indicator | **Prints inner fold progress** (1/3, 2/3, 3/3) |

---

## 11. `evaluate.py` — Multi-Horizon Agreement Filter

| Aspect | Original | Improved |
|--------|----------|----------|
| Multi-horizon filtering | Docstring mentioned it, not implemented | **Implemented** — k=5 LR prediction must agree with k=10 MoE direction for trade execution |
| Dataset path | Hardcoded | Same (would benefit from config, kept simple for course project) |

**How multi-horizon works:** Before executing a Buy/Sell, we check whether the k=5 horizon prediction agrees on direction. If k=10 says "Up" but k=5 says "Down", the trade is downgraded to Hold. This filters out cases where the short-term outlook contradicts the medium-term prediction.

---

## 12. `run_experiment.py` — GPU Info + MoE+BT Stats

- Now prints GPU device name at startup.
- Reports both MoE (no BT) and MoE (with BT) classification metrics.
- Reports win rate and decision accuracy in aggregated results.
- All prints use `flush=True`.

---

## Summary of Expected Impact

| Metric | Before (Fold 1) | Expected Improvement |
|--------|------------------|---------------------|
| MoE Accuracy | ~0.547 | Slight improvement via calibration |
| MoE Macro-F1 | ~0.435 | Improvement from calibrated experts + better gating |
| Cumulative Return (BT) | 0.036 | More trades, potentially higher return |
| Sharpe (BT) | 10.89 (inflated from few trades) | More realistic with more trades |
| Turnover | ~4% | Higher (more trades, better signal utilisation) |
| Profit Factor | 2.34 | Maintained or improved with calibrated decisions |
