import os
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "..", "models", "LogisticRegression.pkl"))

def load_model(path=None):
    path = path or MODEL_PATH
    print("path:", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {path}")
    return model

def validate_input(data):
    if not isinstance(data, list):
        raise ValueError("Input must be a list of feature lists")
    if len(data) == 4 and not isinstance(data[0], list):
        data = [data]
    arr = np.array(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("Each sample must have exactly 4 features")
    return arr
