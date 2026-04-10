import pickle
import glob

for pkl in sorted(glob.glob("results/fold_*/fold_results.pkl")):
    fold = pkl.split('/')[1]
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    print(f"{fold}:")
    print("  y_test shape:", data['y_test'].shape)
    # the backtest results aren't in fold_results.pkl!
    # They are only printed to console!
