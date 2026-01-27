import pickle
import sys
import os

# Add parent directory to path to allow importing core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agents', 'qlearning_weights.pkl')
    with open(path, 'rb') as f:
        weights = pickle.load(f)
        print("Loaded Weights:")
        for k, v in weights.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error loading weights: {e}")
