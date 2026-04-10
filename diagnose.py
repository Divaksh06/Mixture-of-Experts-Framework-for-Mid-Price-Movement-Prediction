"""
Diagnostic script to identify fundamental execution bugs.

Tests:
1. Correlation between signal P(up)-P(down) and actual future return
2. Always-Long baseline
3. Always-Short baseline
4. Always-Flat baseline (should return 0)
"""
import numpy as np
import pickle
from backtest.engine import BacktestEngine

# Load fold 1 results
with open('results/fold_1/fold_results.pkl', 'rb') as f:
    data = pickle.load(f)

pfinal = data['pfinal_test']
y_test = data['y_test']
n_test = len(y_test)

# Reconstruct synthetic mid-price returns (same seed as evaluate.py)
np.random.seed(42 + 1)
mid_price_returns = np.zeros(n_test)
for i in range(n_test):
    if y_test[i] == 0:
        mid_price_returns[i] = abs(np.random.normal(0.00002, 0.00001))
    elif y_test[i] == 2:
        mid_price_returns[i] = -abs(np.random.normal(0.00002, 0.00001))
    else:
        mid_price_returns[i] = np.random.normal(0.0, 0.00005)

# === 1. Signal-Return Correlation ===
signal = pfinal[:, 0] - pfinal[:, 2]
corr = np.corrcoef(signal[:-1], mid_price_returns[1:])[0, 1]
corr_same_t = np.corrcoef(signal, mid_price_returns)[0, 1]
print("=" * 60)
print("  DIAGNOSTIC: Signal-Return Correlation")
print("=" * 60)
print(f"  corr(signal_t, return_{'{t+1}'}):  {corr:.6f}")
print(f"  corr(signal_t, return_t):    {corr_same_t:.6f}")
print()
if corr < 0:
    print("  ⚠️  NEGATIVE CORRELATION → signal is INVERTED!")
    print("  Fix: flip signal = P(down) - P(up)")
elif corr > 0.01:
    print("  ✅ Positive correlation — signal direction is correct")
else:
    print("  ⚠️  Near-zero correlation — signal has no predictive power")

# === 2. Signal vs Label Agreement ===
signal_direction = np.sign(signal)
label_direction = np.zeros(n_test)
label_direction[y_test == 0] = 1.0  # Up
label_direction[y_test == 2] = -1.0  # Down
agreement = np.mean(signal_direction == label_direction)
print(f"\n  Signal-Label agreement rate: {agreement:.4f}")

# === 3. Always-Long Baseline ===
engine_long = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)
engine_long.step('Buy', mid_price_returns[0])
for i in range(1, n_test):
    ret = mid_price_returns[i] if i < n_test - 1 else 0.0
    engine_long.step('Hold', ret)
engine_long.close_position()
long_ret = (engine_long.portfolio_value - 10000.0) / 10000.0

# === 4. Always-Short Baseline ===
engine_short = BacktestEngine(initial_capital=10000.0, transaction_cost=0.0001)
engine_short.step('Sell', mid_price_returns[0])
for i in range(1, n_test):
    ret = mid_price_returns[i] if i < n_test - 1 else 0.0
    engine_short.step('Hold', ret)
engine_short.close_position()
short_ret = (engine_short.portfolio_value - 10000.0) / 10000.0

print()
print("=" * 60)
print("  BASELINE STRATEGIES")
print("=" * 60)
print(f"  Always-Long return:  {long_ret:.6f}")
print(f"  Always-Short return: {short_ret:.6f}")
print(f"  Model return (bt):   -0.389396")
print()
if long_ret > 0 and short_ret > 0:
    print("  ⚠️  BOTH baselines positive → execution engine is broken")
elif -0.389396 < min(long_ret, short_ret):
    print("  ⚠️  MODEL IS WORSE THAN BOTH BASELINES → execution is broken")
else:
    print("  ✅ Model return is between baselines")

# === 5. Cost analysis ===
c_total = 0.0001 + 0.00015
avg_return_magnitude = np.mean(np.abs(mid_price_returns))
print()
print("=" * 60)
print("  COST ANALYSIS")
print("=" * 60)
print(f"  Cost per side:           {c_total:.6f}")
print(f"  Cost per round-trip:     {2*c_total:.6f}")
print(f"  Avg |return| per step:   {avg_return_magnitude:.6f}")
print(f"  Steps to break even:     {2*c_total / avg_return_magnitude:.1f}")
print(f"  Current avg hold time:   14.6 steps")
break_even = 2 * c_total / avg_return_magnitude
if 14.6 < break_even:
    print(f"  ⚠️  avg_hold ({14.6}) < break_even ({break_even:.1f}) → COST DESTROYS EDGE")
