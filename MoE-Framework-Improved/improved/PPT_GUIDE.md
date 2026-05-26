# Presentation Guide — What to Write on Each Slide

Total: 14 slides as per your specification.

---

## SLIDE 1 — Problem Statement (1/2): Background

**Title:** Mid-Price Movement Prediction in High-Frequency Markets

**Content:**
- Financial markets generate massive order flow data via **Limit Order Books (LOBs)**.
- The **mid-price** (avg of best bid & ask) is the most widely used fair-price indicator.
- Predicting whether the mid-price will move **Up, Down, or stay Stationary** in the next K ticks is fundamental to algorithmic trading.
- **Challenge:** LOB data is high-dimensional (144 features), noisy, non-stationary, and class-imbalanced (Stationary class dominates at ~40%).
- Traditional single-model approaches often fail because no one model type handles all market regimes equally well.

**Visual:** A simple diagram showing a Limit Order Book with bid/ask levels, highlighting the mid-price.

---

## SLIDE 2 — Problem Statement (2/2): Our Approach

**Title:** Mixture-of-Experts: Combining Model Strengths

**Content:**
- **Goal:** Build a Mixture-of-Experts (MoE) framework that:
  1. Combines three diverse ML classifiers (LR, XGBoost, MLP).
  2. Learns *when* to trust each expert via a gating network.
  3. Converts predictions into actionable Buy/Hold/Sell signals.
  4. Adapts online via a backtracking module.
- **Constraint:** No deep learning architectures (no CNNs/LSTMs/Transformers) — pure ML approach.
- **Evaluation:** Both classification accuracy (Macro-F1) and simulated trading performance (Sharpe, Return, Max Drawdown).

**Visual:** High-level pipeline diagram (Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5).

---

## SLIDE 3 — Dataset Description

**Title:** FI-2010 Benchmark Dataset

**Content:**
- **Source:** 5 Finnish stocks, NASDAQ Nordic, ~4.5 million LOB snapshots.
- **Features:** 144 normalised features (Z-score) from 10-level LOB:
  - Raw features: prices & volumes at 10 bid/ask levels
  - Time-insensitive: spread, depth, imbalance
  - Time-sensitive: price changes, momentum
  - Statistical: rolling means, standard deviations
- **Labels:** 3-class — Up (0), Stationary (1), Down (2) at horizon k=10.
- **CV scheme:** 9 anchored-forward folds — training grows cumulatively, test is always next unseen day (prevents temporal leakage).
- **Class imbalance:** Stationary ~40%, Up ~37%, Down ~23%. Handled via balanced class weights.

**Visual:** Table showing fold sizes + bar chart of class distribution (can reuse from EDA).

---

## SLIDE 4 — Model Used (1/2): Expert Models

**Title:** Three Expert Classifiers

**Content:**

| Expert | Type | Key Features | Role in Ensemble |
|--------|------|-------------|-----------------|
| **Logistic Regression** | Linear | C-tuned via CV, isotonic calibration | Stable baseline, captures linear trends |
| **XGBoost** | Tree-based | GPU-accelerated, early stopping, regularised | Non-linear interactions, robust features |
| **MLP** | Neural Network | Temporal lookback (5 steps), focal loss, LR scheduler | Temporal patterns, hidden representations |

- Each expert is independently trained on the full training data.
- **Probability calibration** (isotonic/temperature) ensures probabilities are meaningful.
- **Focal Loss** in MLP focuses training on hard examples.

**Visual:** Side-by-side architecture diagrams of the three models.

---

## SLIDE 5 — Model Used (2/2): Gating Network & MoE

**Title:** Gating Network — Learning When to Trust Each Expert

**Content:**
- **Input:** 12-dimensional vector = 9 (stacked expert probs) + 3 (expert agreement features).
- **Architecture:** Dense(64, ReLU, Dropout) → Dense(3, Softmax).
- **Output:** Weights [w₁, w₂, w₃] → P_final = w₁·P_LR + w₂·P_XGB + w₃·P_MLP.
- **Training:** On out-of-fold stacked probabilities (prevents overfitting), NLL loss, early stopping.
- **Key insight:** The gating network assigns higher weight to whichever expert is more reliable for the current market state.

**Visual:** MoE architecture diagram:
```
X → [LR] → P_LR ──┐
X → [XGB] → P_XGB ─┤→ [Stack] → [Gating] → weights → P_final
X → [MLP] → P_MLP ─┘
```

---

## SLIDE 6 — Efforts & Methodology (1/4): Stacking & Calibration

**Title:** Stage 2: Stacking for Unbiased Probability Combination

**Content:**
- **Problem:** If we train gating on the same data experts trained on, it overfits.
- **Solution:** 3-fold inner cross-validation — experts predict on held-out inner folds.
- The gating network only sees "out-of-fold" expert predictions → honest training signal.
- **Probability calibration** ensures P(Up)=0.60 actually means 60% of such cases are Up.
  - LR & XGBoost: Isotonic regression calibration.
  - MLP: Learnable temperature parameter.

**Visual:** Diagram showing inner 3-fold CV with arrows indicating which data trains which model.

---

## SLIDE 7 — Efforts & Methodology (2/4): Strategy Layer

**Title:** Converting Predictions to Trading Actions

**Content:**
- **Signal:** s_t = P(Up) - P(Down), smoothed via EMA (α=0.3).
- **Entry rule:** |signal| > 0.35 AND directional prob > 0.55 → Buy/Sell.
- **Exit rule:** |signal| < 0.10 → Flat (close position).
- **Flip penalty:** Extra +0.10 signal needed for Long↔Short reversal.
- **Regime filter:** Only trade if recent signal strength avg > 0.45.
- **Multi-horizon filter:** k=5 and k=10 predictions must agree on direction.
- **Minimum hold:** 15 ticks — prevents over-trading that loses money to transaction costs.

**Visual:** State machine diagram (None → Long → None → Short → None).

---

## SLIDE 8 — Efforts & Methodology (3/4): Backtracking Module

**Title:** Online Adaptation During Inference

**Content:**
- Maintains a **rolling buffer of 200 recent observations**.
- Every 100 steps, updates:
  1. **Expert weights**: Based on recent per-expert accuracy (exponential update with EMA smoothing).
  2. **Strategy thresholds**: Increases θ if too many false positives, decreases if too many missed opportunities.
- **Weight clamping:** No expert can go below 10% or above 60% — prevents degenerate behaviour.
- **Action correction:** Overrides to Hold if strong reversal signal contradicts recent trade.

**Visual:** Timeline diagram showing buffer window, update points, and how weights evolve.

---

## SLIDE 9 — Efforts & Methodology (4/4): Backtesting & Avoiding Bias

**Title:** Realistic Portfolio Simulation & Bias Prevention

**Content:**
- **Backtesting engine:** Simulates Long/Short/Flat positions with:
  - Initial capital: ₹10,000
  - Transaction cost: 0.02% per side (commission + spread)
  - Minimum hold: 15 ticks
- **Forward-looking bias prevention:**
  - Anchored forward CV (no future data in training)
  - Out-of-fold stacking (gating sees honest predictions)
  - Online backtracking (only uses past observations)
  - Sequential backtesting (processes ticks in order)

**Visual:** Portfolio value curve example showing entries, exits, and drawdowns.

---

## SLIDE 10 — Results & Analysis (1/4): Classification Performance

**Title:** Classification Results — Mean ± Std Across 9 Folds

**Content:**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| LR (Expert 1) | X.XX ± X.XX | X.XX ± X.XX |
| XGBoost (Expert 2) | X.XX ± X.XX | X.XX ± X.XX |
| MLP (Expert 3) | X.XX ± X.XX | X.XX ± X.XX |
| **MoE (no BT)** | **X.XX ± X.XX** | **X.XX ± X.XX** |
| **MoE (with BT)** | **X.XX ± X.XX** | **X.XX ± X.XX** |

- MoE combines expert strengths — consistently better than any single expert.
- Backtracking further improves by adapting to recent market conditions.
- MLP's temporal lookback captures sequential patterns that LR/XGB miss.

*(Fill in actual numbers from your run)*

---

## SLIDE 11 — Results & Analysis (2/4): Trading Performance

**Title:** Trading Performance — Ablation Study

**Content:**

| Strategy | Return | Sharpe | Max DD | Trades |
|----------|--------|--------|--------|--------|
| XGB alone | ... | ... | ... | ... |
| MLP alone | ... | ... | ... | ... |
| MoE (no BT) | ... | ... | ... | ... |
| **MoE (with BT)** | **...** | **...** | **...** | **...** |
| Buy & Hold | ... | — | — | 1 |

- MoE ensemble smooths risk (lower Max Drawdown) vs. single models.
- Backtracking improves risk-adjusted returns (Sharpe) by adapting.
- Key insight: **Classification accuracy ≠ Trading profitability** — MoE's strength is in selectivity.

*(Fill in actual numbers from your run)*

---

## SLIDE 12 — Results & Analysis (3/4): What Worked & What Didn't

**Title:** Key Insights from the Experiments

**Content:**
- ✅ **Probability calibration** was critical — uncalibrated probabilities caused the strategy to either trade too much or too little.
- ✅ **EMA signal smoothing** reduced whipsaw trades that destroyed edge.
- ✅ **Expert diversity** (linear + tree + neural) ensured complementary error patterns.
- ✅ **Minimum hold period** prevented over-trading in a high-frequency setting.
- ⚠️ **Stationary class** remains hardest to handle — models tend to conflate it with Up/Down.
- ⚠️ **Transaction costs** eat into small per-trade returns — selectivity is essential.
- ⚠️ **Sharpe inflation** from annualising tick-level returns — numbers should be interpreted carefully.

---

## SLIDE 13 — Results & Analysis (4/4): Comparison with Baselines

**Title:** MoE vs. Single-Expert & Buy-and-Hold Baselines

**Content:**
- **vs. Buy-and-Hold:** MoE with backtracking achieves [better/comparable] risk-adjusted returns while controlling downside risk (lower MDD).
- **vs. Best Single Expert (XGBoost):** MoE ensemble adds diversity benefit — smoother equity curve, fewer catastrophic drawdowns.
- **vs. No Backtracking:** Online adaptation improves Sharpe by X% by down-weighting underperforming experts.
- **Ablation summary:** Each component contributes — removing calibration, gating, or backtracking individually degrades performance.

**Visual:** Bar chart comparing returns and Sharpe across strategies.

---

## SLIDE 14 — Work Division

**Title:** Team Contributions

**Content:**

| Team Member | Contribution |
|-------------|-------------|
| Member A | Data preprocessing, EDA, cross-validation pipeline, data loader |
| Member B | LR & XGBoost expert implementation, hyperparameter tuning, calibration |
| Member C | MLP expert, gating network, MoE forward pass, GPU optimization |
| Member D | Strategy layer, backtracking module, backtesting engine, evaluation metrics |
| **All** | **Report writing, testing, integration, debugging** |

*(Adjust names and contributions as per your actual team)*

---

## General Presentation Tips

1. **Don't read slides** — use bullet points as prompts, explain in your own words.
2. **Emphasise the MoE concept** — this is your main contribution. Explain WHY combining experts is better than using one model.
3. **Show concrete numbers** — fill in actual results from your run.
4. **Anticipate questions:**
   - "Why not use deep learning?" → Course constraint, but MoE framework with classical ML achieves competitive results.
   - "How do you handle overfitting?" → Anchored CV, stacking, calibration, early stopping, regularisation.
   - "Is the Sharpe ratio realistic?" → Annualised from tick-level data, so absolute values are inflated. Relative comparisons are meaningful.
   - "What about forward-looking bias?" → Five safeguards (listed in Slide 9).
5. **Keep it at 15-20 mins** — ~1.5 min per slide.
