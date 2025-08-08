# src/utils.py
import os
import pickle
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/LogisticRegression.pkl")

def load_model(path=None):
    path = path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def validate_input(data):
    # expecting a list or nested list of shape (n_samples, 4)
    if not isinstance(data, list):
        raise ValueError("Input must be a list of feature lists")
    # Single sample convenience: allow [f1, f2, f3, f4]
    if len(data) == 4 and not isinstance(data[0], list):
        data = [data]
    arr = np.array(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("Each sample must have 4 features")
    return arr
