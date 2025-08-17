import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_COLS_PATH = os.path.join(PROJECT_ROOT, "results", "train_columns.txt")

quant_columns = [
    "LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
    "TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","GarageArea","WoodDeckSF",
    "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"
]

def apply_notebook_fillna(df: pd.DataFrame) -> pd.DataFrame:
    numerical_columns=df.select_dtypes(include=[np.number]).columns
    numerical_columns_list=numerical_columns.tolist()
    for i in numerical_columns_list:
        if i!='GarageYrBlt':
            df[i]=df[i].fillna(df[i].median())
        else:
            df[i]=df[i].fillna(0)
    
    categorical_columns=df.select_dtypes(include=object).columns
    categorical_columns_list=categorical_columns.tolist()
    for i in categorical_columns:
        if i=="Alley":
            df["Alley"] = df["Alley"].fillna("No Alley")
        if i=="MasVnrType":
            df["MasVnrType"] = df["MasVnrType"].replace("None", "No Masonry Veneer")
            df['MasVnrType'] = df['MasVnrType'].fillna('No Masonry Veneer')
        if i in ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]:
            df[i] = df[i].fillna("No Basement")
        if i=="FireplaceQu":
            df["FireplaceQu"] = df["FireplaceQu"].fillna("No Fireplace")
        if i in ["GarageType","GarageFinish","GarageQual","GarageCond"]:
            df[i] = df[i].fillna("No Garage")
        if i=="PoolQC":
            df["PoolQC"] = df["PoolQC"].fillna("NoPool")
        if i=="Fence":
            df["Fence"] = df["Fence"].fillna("No Fence")
        if i=="MiscFeature":
            df["MiscFeature"] = df["MiscFeature"].fillna("No Miscellaneous feature")
        else:
            df[i]=df[i].fillna(df[i].mode()[0])
        # other fillna performed in your notebook can be added here following the same pattern
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    for c in quant_columns:
        if c in df.columns:
            df[c] = np.log1p(df[c].fillna(0))
    if "SalePrice" in df.columns:
        df["SalePrice"]=np.log1p(df["SalePrice"])
    return df

def fit_encode_train(df: pd.DataFrame):
    df = apply_notebook_fillna(df.copy())
    df = log_transform(df)
    # separate y if present
    y = None
    if "SalePrice" in df.columns:
        y = df["SalePrice"].copy()
        X = df.drop(columns=["SalePrice"])
    else:
        X = df.copy()
    if "Id" in X.columns:
        X=X.drop(columns=["Id"])
    # one-hot encode categoricals 
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    dummies = pd.get_dummies(X[cat_cols], drop_first=False) if cat_cols else pd.DataFrame()
    if not dummies.empty:
        X_out = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    else:
        X_out = X.copy()
    # save columns for later alignment
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    with open(TRAIN_COLS_PATH, "w", encoding="utf-8") as f:
        for c in X_out.columns:
            f.write(c + "\n")
    return X_out, y

def encode_test_like_train(df: pd.DataFrame):
    """Apply notebook preprocessing to test/unseen data and align to train columns saved earlier."""
    df = apply_notebook_fillna(df.copy())
    df = log_transform(df)
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    dummies = pd.get_dummies(df[cat_cols], drop_first=False) if cat_cols else pd.DataFrame()
    if not dummies.empty:
        X_test = pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    else:
        X_test = df.copy()
    # align columns to train
    if os.path.exists(TRAIN_COLS_PATH):
        with open(TRAIN_COLS_PATH, "r", encoding="utf-8") as f:
            train_cols = [l.strip() for l in f.readlines()]
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
    return X_test

