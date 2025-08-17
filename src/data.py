import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def load_train(path=None):
    path = path or os.path.join(DATA_DIR, "train.csv")
    return pd.read_csv(path)

def load_test(path=None):
    path = path or os.path.join(DATA_DIR, "test.csv")
    return pd.read_csv(path)

def save_processed(df, filename="train_processed.csv"):
    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, filename)
    df.to_csv(out, index=False)
    return out
