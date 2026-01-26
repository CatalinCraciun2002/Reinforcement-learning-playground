
import pickle
import sys

try:
    with open('qlearning_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
        print("Loaded Weights:")
        for k, v in weights.items():
            print(f"  {k}: {v}")
except Exception as e:
    print(f"Error loading weights: {e}")
