# Glossary of Technical Terms — ML & Quant Finance

This document explains every technical term used in the project. Organised by category for quick reference.

---

## A. Machine Learning — Core Concepts

### Logistic Regression (LR)
A linear classification model that predicts probabilities using the logistic (sigmoid) function. For multi-class problems, it uses the **softmax function** (multinomial LR). Despite its simplicity, it works well when the decision boundary is approximately linear. The `C` parameter controls **regularisation strength** (smaller C = stronger regularisation).

### XGBoost (eXtreme Gradient Boosting)
A gradient-boosted decision tree algorithm. It builds an ensemble of shallow decision trees **sequentially** — each new tree corrects the errors of the previous ones. Key hyperparameters:
- **max_depth**: Maximum depth of each tree (deeper = more complex patterns, risk of overfitting).
- **learning_rate**: How much each new tree contributes (lower = slower but more robust learning).
- **n_estimators**: Maximum number of trees to build.
- **early stopping**: Stops adding trees when validation performance stops improving.

### Multi-Layer Perceptron (MLP)
A feedforward neural network with one or more hidden layers. Each layer performs: output = activation(W·input + b). Our MLP uses:
- **ReLU activation**: max(0, x) — simple, effective, avoids vanishing gradients.
- **Batch Normalisation**: Normalises layer inputs to zero mean, unit variance — stabilises and speeds up training.
- **Dropout**: Randomly zeroes a fraction of neurons during training — prevents overfitting by forcing the network to learn redundant representations.

### Mixture-of-Experts (MoE)
An ensemble technique where multiple "expert" models specialise in different parts of the input space, and a **gating network** learns to combine their outputs adaptively. Unlike simple voting (which averages uniformly), MoE assigns **input-dependent weights** to each expert.

### Gating Network
A small neural network that takes the experts' probability outputs (concatenated) as input and outputs a weight vector [w₁, w₂, w₃] via softmax. The final prediction is P_final = w₁·P_LR + w₂·P_XGB + w₃·P_MLP. The gating network learns **when** to trust which expert.

### Stacking (Stacked Generalisation)
A meta-learning technique where a second-level model is trained on the **out-of-fold predictions** of the base models, rather than on the original features. This prevents the gating network from overfitting to the training data. We use 3-fold inner CV to generate these out-of-fold probabilities.

---

## B. Machine Learning — Training Concepts

### Cross-Validation (CV)
A technique to evaluate model performance by splitting data into multiple train/test folds. **Stratified K-Fold** ensures each fold has the same class distribution as the full dataset. **Anchored Forward CV** (used here) is specific to time-series: training data grows cumulatively while test is always the next time period.

### Early Stopping
A regularisation technique that monitors a validation metric during training and stops when performance stops improving. The **patience** parameter specifies how many epochs to wait before stopping. Prevents overfitting by not training longer than necessary.

### Learning Rate Scheduler (ReduceLROnPlateau)
Automatically reduces the learning rate when a monitored metric (e.g., validation F1) stops improving. This allows the model to initially make large updates, then fine-tune with smaller steps as it approaches a good solution.

### Focal Loss
A modified cross-entropy loss that **down-weights easy examples** and focuses training on hard-to-classify samples. Controlled by parameter γ (gamma): higher γ = more focus on hard examples. Particularly useful for class-imbalanced datasets. Formula: FL(p_t) = -(1-p_t)^γ · log(p_t).

### Probability Calibration
Post-training adjustment of predicted probabilities so they match actual event frequencies. If a model says P(Up)=0.70, calibration ensures that ~70% of such predictions actually are Up. Methods:
- **Isotonic Regression**: Non-parametric — fits a non-decreasing function to map raw probabilities to calibrated ones.
- **Temperature Scaling**: Divides logits by a learnable temperature T before softmax. T>1 softens probabilities, T<1 sharpens them.

### Class Weights (Balanced)
When classes are imbalanced (e.g., Stationary is 40% of data), the loss function can be re-weighted so that minority classes contribute more. Computed as: w_c = N / (K × n_c), where N=total samples, K=num classes, n_c=samples in class c.

### Gradient Clipping
Limits the magnitude of gradients during backpropagation. Prevents **gradient explosion** (very large updates that destabilise training). We clip to max norm = 1.0.

### Weight Decay (L2 Regularisation)
Adds a penalty term λ·‖W‖² to the loss function, encouraging smaller weights. Prevents overfitting by discouraging the model from relying too heavily on any single feature.

### Macro-F1 Score
The unweighted average of F1 scores for each class. F1 = 2·(precision × recall)/(precision + recall). Macro-F1 treats all classes equally regardless of their size, making it a better metric than accuracy for imbalanced datasets.

### Softmax Function
Converts a vector of raw scores (logits) into a probability distribution: softmax(z_i) = exp(z_i) / Σ_j exp(z_j). All outputs sum to 1.0 and are in [0, 1].

---

## C. Machine Learning — Meta-Features

### Expert Agreement Features
For each class c, we compute the standard deviation of expert predictions: std(P_LR(c), P_XGB(c), P_MLP(c)). When experts agree (low std), predictions are more reliable. When they disagree (high std), the gating network should be more cautious. These 3 additional features enrich the gating network's input.

---

## D. Quantitative Finance — Market Concepts

### Limit Order Book (LOB)
A real-time record of all outstanding buy and sell orders for a financial instrument at various price levels. The "10-level" LOB shows the 10 best bid and 10 best ask prices and their volumes.

### Mid-Price
The average of the best bid and best ask prices: mid = (best_ask + best_bid) / 2. Represents the "fair" price at any moment.

### Bid-Ask Spread
The difference between the best ask (lowest sell price) and best bid (highest buy price): spread = ask - bid. Represents the cost of immediate execution. Tighter spread = more liquid market.

### Order Imbalance
The relative difference between bid and ask volumes: OI = (V_bid - V_ask) / (V_bid + V_ask). Positive OI suggests buying pressure (potential price increase).

### Transaction Costs
Costs incurred when executing a trade:
- **Commission**: Fixed fee per trade (we use 1 basis point = 0.01%).
- **Spread cost**: Half the bid-ask spread, paid on each side of a trade (we use 1 basis point).
- **Round-trip cost**: Total cost to open and close a position = 2 × (commission + spread).

### Basis Point (bp)
1 basis point = 0.01% = 0.0001. A common unit for expressing small percentages in finance.

---

## E. Quantitative Finance — Strategy Concepts

### Signal (P(Up) − P(Down))
A scalar measure of directional conviction. Positive signal = model predicts upward movement. The magnitude indicates confidence.

### Tau Entry (τ_entry) / Tau Exit (τ_exit)
Threshold values for the signal. A trade is only entered when |signal| > τ_entry (strong conviction) and exited when |signal| < τ_exit (conviction fades). The gap between τ_entry and τ_exit creates a **hysteresis band** that prevents rapid oscillation between positions.

### Hysteresis Band
A buffer zone between entry and exit thresholds. Once a position is entered at τ_entry, it is maintained until signal drops to τ_exit (which is much lower). This prevents **churn** (rapid buy/sell/buy/sell).

### Flip Penalty
Extra signal strength required to reverse a position (e.g., from Long to Short). This penalises rapid direction changes, which are costly due to double transaction costs.

### Regime Filter
A check that recent market conditions show sufficient predictability before trading. We compute the average of max(P(Up), P(Down)) over the last 10 steps. If this average is below 0.45, the market is considered "uncertain" and no new trades are entered.

### EMA (Exponential Moving Average) Smoothing
A technique to smooth noisy signals. The smoothed signal is: s_ema(t) = α·s(t) + (1−α)·s_ema(t−1). Higher α gives more weight to recent values (more responsive, more noisy). Lower α gives more weight to history (smoother, more lag).

---

## F. Quantitative Finance — Backtracking Concepts

### Online Adaptation
Adjusting model parameters during inference (test time) based on observed outcomes. Unlike re-training, this uses a lightweight buffer-based mechanism.

### Rolling Buffer
A fixed-size deque (double-ended queue) that stores the most recent N observations. When full, adding a new observation automatically discards the oldest one. Ensures adaptation is based on recent, relevant data.

### Exponential Weight Update
$w'_i = w_i \cdot \exp(\eta \cdot \alpha_i) / \text{normalisation}$. Experts with higher recent accuracy get higher weights. The exponential ensures smooth, multiplicative adjustment.

### False Positive Rate (FPR)
For a "Buy" action: the fraction of Buy actions where the true label was NOT Up. High FPR means the strategy is buying too aggressively — response: increase θ_buy.

### False Negative Rate (FNR)
For "Buy": the fraction of true Up events where the action was NOT Buy. High FNR means the strategy is missing opportunities — response: decrease θ_buy.

---

## G. Quantitative Finance — Performance Metrics

### Cumulative Return
Total percentage gain/loss: R = (V_final − V_initial) / V_initial. A return of 0.05 = 5% gain.

### Sharpe Ratio
Risk-adjusted return: Sharpe = (mean(returns) / std(returns)) × √(periods_per_year). Higher = better. Annualised using N_ann = 252 × 390 for intraday tick data. A Sharpe above 1.0 is generally considered good for daily strategies.

### Maximum Drawdown (MDD)
The largest peak-to-trough decline in portfolio value: MDD = max_t [(peak_t − V_t) / peak_t]. Represents the worst-case loss an investor would have experienced. Lower is better.

### Win Rate
Fraction of completed trades that were profitable. Win rate alone is not informative — a strategy can have 90% win rate but still lose money if the 10% losses are very large.

### Profit Factor
Gross Profit / Gross Loss. A profit factor of 2.0 means the strategy earned twice as much on winning trades as it lost on losing trades. Above 1.0 is profitable.

### Decision Accuracy
Fraction of Buy/Sell actions that matched the true price direction. Buy when price went Up = correct. Sell when price went Down = correct. Ignores Hold actions.

### Turnover
Fraction of time steps with active trade decisions (Buy, Sell, or Flat). High turnover = frequent trading = higher transaction costs. Low turnover = infrequent trading = may miss opportunities.

### Buy-and-Hold Return
Benchmark return from simply buying at the start and holding throughout the test period. Any active strategy should aim to beat this baseline.

---

## H. Quantitative Finance — Backtesting Concepts

### Backtesting
Simulating a trading strategy on historical data to evaluate its hypothetical performance. Critical rules:
1. No future information at any time step.
2. Realistic transaction costs.
3. Position sizing constraints.

### Long Position
Buying an asset and profiting when its price increases. Return = (P_sell − P_buy) / P_buy.

### Short Position
Selling a borrowed asset and profiting when its price decreases. Return = (P_sell − P_buy) / P_sell (inverted sign).

### Minimum Holding Period (hold_k)
A constraint that forces the strategy to hold a position for at least K ticks after opening. Prevents excessively frequent trading that would be eaten by transaction costs.

### Forward-Looking Bias (Look-Ahead Bias)
Using information that would not have been available at the time of the trading decision. Common sources: using future labels for feature engineering, computing statistics over the full dataset, or peeking at test data during training. Our pipeline is designed to avoid this at every stage.
