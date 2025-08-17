'''import os, joblib
import pandas as pd
import numpy as np
from src.preprocess import encode_test_like_train

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "final_model.pkl")

def load_model(path=None):
    path = path or MODEL_PATH
    return joblib.load(path)

def predict_raw_df(df: pd.DataFrame, model):
    X_test = encode_test_like_train(df)
    preds = model.predict(X_test)
    preds = np.expm1(preds)
    return preds'''

# src/predict.py  --- replace predict_raw_df with this implementation
import os
import joblib
import numpy as np
import pandas as pd
import re
from src.preprocess import encode_test_like_train

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MODEL_PATH = os.path.join(RESULTS_DIR, "final_model.pkl")
TRAIN_COLS_PATH = os.path.join(RESULTS_DIR, "train_columns.txt")

def load_model(path=None):
    path = path or MODEL_PATH
    return joblib.load(path)

def _normalize_name(s: str) -> str:
    """Normalize a feature name for fuzzy matching: lower + remove non-alphanum."""
    if s is None:
        return ""
    return re.sub(r'[^0-9a-z]', '', str(s).lower())

def _get_model_feature_names(model):
    """Return the feature names expected by the model; fallback to train_columns.txt file."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # fallback: try to read results/train_columns.txt
    if os.path.exists(TRAIN_COLS_PATH):
        with open(TRAIN_COLS_PATH, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    raise RuntimeError("Model missing feature_names and no train_columns.txt found.")

def _align_and_rename_columns(X: pd.DataFrame, model_feature_names: list) -> pd.DataFrame:
    """
    Align X's columns to exactly match model_feature_names.
    Strategy:
    - Build normalized map of X columns (normalize name -> original column).
    - For each model_feature:
        - If an identical column exists in X -> keep it.
        - Else if a normalized match exists -> rename that X column to the model_feature name.
        - Else -> create model_feature column filled with 0.
    - Finally, reindex X to the exact ordered list model_feature_names (fill missing with 0).
    - Drop any columns in X that are not in model_feature_names.
    """
    X = X.copy()
    orig_cols = list(X.columns)
    norm_map = { _normalize_name(c): c for c in orig_cols }

    # rename mapping
    rename_map = {}
    for mf in model_feature_names:
        norm = _normalize_name(mf)
        if mf in X.columns:
            continue  # exact match present
        if norm in norm_map:
            # rename the matched existing col to the model's exact name
            existing_col = norm_map[norm]
            if existing_col != mf:
                rename_map[existing_col] = mf

    if rename_map:
        X = X.rename(columns=rename_map)

    # ensure all model features exist; fill with 0 if missing
    for mf in model_feature_names:
        if mf not in X.columns:
            X[mf] = 0

    # finally reindex to model order (this drops extras)
    X = X.reindex(columns=model_feature_names, fill_value=0)
    return X

def predict_raw_df(df: pd.DataFrame, model):
    X_test = encode_test_like_train(df)
    model_feat_names = _get_model_feature_names(model)
    X_aligned = _align_and_rename_columns(X_test, model_feat_names)
    preds_log = model.predict(X_aligned)
    preds = np.expm1(preds_log)
    return preds


