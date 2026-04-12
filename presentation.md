# Mixture of Experts Framework for Mid-Price Movement Prediction
## PRML Capstone Presentation Outline

---

## Slide 1: Title Slide
- **Title:** Mixture-of-Experts Framework for High-Frequency Mid-Price Movement Prediction
- **Course:** Pattern Recognition and Machine Learning (PRML)
- **Team Name / Members**
- **Core Concept:** Predicting Limit Order Book trajectories using a multi-algorithm neural ensemble.

---

## Slide 2: The Problem Statement
- **The Domain:** High-Frequency Trading (HFT) and Limit Order Books (LOB).
- **The Challenge:** The market is an intensely noisy, non-stationary environment. Prices flicker back and forth due to microstructure noise, making standard regression models fail.
- **The Classifier Paradox:** When trying to predict if a price goes Up, Down, or stays Stationary, standard Machine Learning models (like Neural Networks) optimize for total accuracy by simply predicting "Stationary" almost all the time. 
- **The Result:** They get high accuracy but lose money because they fail to capture actual directional trends.

---

## Slide 3: Our Proposed Solution (MoE Architecture)
- Instead of using one model, we built a **Mixture-of-Experts (MoE)**.
- **Why?** Different models learn different things. We combined:
  1. **Logistic Regression (Expert 1):** Fast, linear baseline.
  2. **XGBoost (Expert 2):** Aggressive, non-linear decision trees excellent at spatial tabular splits.
  3. **Temporal MLP (Expert 3):** A PyTorch Neural Network explicitly designed to capture historical time sequences.
- An overarching **Gating Network** learns which expert to trust given the current state of the order book dynamically.

---

## Slide 4: Dataset & The DecPre Advantage
- **Dataset:** FI-2010 Benchmark Dataset (4 million limit order events across 10 days).
- **Our Strategic Choice:** The dataset provides pre-normalized "Z-Scored" files, but we **expressly forced the raw Decimal Precision (DecPre) files**.
- **Reason:** In physical markets, order volume cannot be negative. If we used Z-score data, negative volumes would have collapsed our fractional non-linear formulas (like Order Imbalance) into infinity or flipped the signs. DecPre allowed us to maintain mathematical purity.

---

## Slide 5: Feature Engineering & Preprocessing
- After Exploratory Data Analysis (EDA), we natively constructed 3 explicit high-conviction features on top of the original 144:
  1. **Level-1 Spread:** $Ask_1 - Bid_1$
  2. **Level-1 Imbalance:** $(BidVol - AskVol) / (BidVol + AskVol)$
  3. **Level-5 Imbalance:** Accumulating deeper outer limits of the order book to remove flicker noise.
- This immediately gave our linear models (Logistic Regression) direct access to powerful non-linear fractional relationships they couldn't naturally compute.

---

## Slide 6: The Temporal MLP Network
- While XGBoost digested the flat 147 features brilliantly, neural networks require sequence bias.
- We passed the 147 features into a rolling memory stack (`lookback = 5`).
- **Input Dimension:** $147 * 5 = 735$ dimensions.
- **Loss Function:** We deployed a Focal Loss function ($\gamma=3.0$) to punish the neural network when it made lazy predictions on the majority "Stationary" class, forcing it to find the hard directional patterns.

---

## Slide 7: Algorithmic Trading Layer (The State Machine)
- Machine Learning probabilities need to be converted into physical trades, factoring in transaction costs.
- **Hysteresis Band:** We don't trade unless the predictive probability difference breaches $\tau_{entry} = 0.60$.
- **Latency Hold:** Every action is hard-locked for $hold_k=25$ updates to conservatively simulate crossing the spread.
- **The Regime Filter:** We built a 20-step moving average queue that shuts down all market entries when the model's rolling confidence average drops below $0.55$. This explicitly prevents the model from trading during sideways "chop" traps.

---

## Slide 8: The Experiment (Anchored Forward Cross-Validation)
- We evaluated the models securely using Anchored Forward CV across 9 chronological folds.
- This represents real life perfectly: Day 1 trains Day 2. Days 1&2 train Day 3. 
- *Data leakage is strictly impossible in this setup.*

---

## Slide 9: Results - The Ultimate Predictor Paradox
- **Graphic Idea:** Show a table comparing Classification Accuracy to Trading Returns.
- **Temporal MLP:** Achieved the absolute highest Classification Macro-F1 (**56.2%**)... but lost almost all profitability due to overly conservative probability distributions.
- **XGBoost:** Had lower accuracy (**50.1% F1**) but was highly decisive, capturing **+20.9%** return.
- **The Takeaway:** The PyTorch MLP maximized Cross-Entropy accuracy by conservatively guessing "Stationary," rendering it a poor standalone trading bot. XGBoost took actionable directional risks. (Spearman correlation $\rho = -0.046$, proving accuracy and trading profit are completely decoupled).

---

## Slide 10: Conclusion - The MoE Result (Risk vs Reward)
- The Mixture-of-Experts pipeline learned to combine the models. Rather than just maximizing profit, the Gating Network explicitly minimized risk.
- **Risk Mitigation:** Gated MoE produced an essentially identical return to XGBoost (**~20.0%**), but effectively cut out downside volatility, boosting the **Sortino Ratio by 25%** (from 0.096 to 0.122) and the **Return-to-Volatility ratio by 35%**.
- **Ablation Insight:** Simple averaging/weighting completely destroys the algorithmic edge (0.4% return). Dynamic gating is strictly required to restore profitability (~50x better than simple ensembles).
- **Latency Optimization:** An experiment feeding 147 raw Limit Order Book features into the router alongside the experts yielded no additional alpha. The base experts fully exhaust the LOB variance. By isolating the gating to just 3 probabilities, the MoE executes in microseconds, an absolute necessity for real-world HFT.
