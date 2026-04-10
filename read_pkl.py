import sys
import pickle

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_pkl.py <path_to_pkl_file>")
        print("Example: python read_pkl.py results/fold_1/fold_results.pkl")
        return
    
    file_path = sys.argv[1]
    print(f"Loading {file_path}...\n")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
        
    print("=== Keys Inside Pickle File ===")
    for key in data.keys():
        val = data[key]
        t = type(val).__name__
        print(f"- {key:<20} (Type: {t})")
        
    print("\n=== Model Metrics Summary ===")
    if 'moe_accuracy' in data:
        print(f"MoE Accuracy: {data['moe_accuracy']:.4f}")
    if 'moe_macro_f1' in data:
        print(f"MoE Macro F1: {data['moe_macro_f1']:.4f}")
        
    print("\nIf you want to load this yourself in a Notebook/Python shell:")
    print(f">>> import pickle")
    print(f">>> with open('{file_path}', 'rb') as f:")
    print(f">>>     data = pickle.load(f)")

if __name__ == '__main__':
    main()
