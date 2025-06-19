from sklearn.model_selection import train_test_split
from models.linear_regression import LinearReg
from src.metrics import*

def split_dataset(df, target_column="Precio_usd", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# normalize_utils.py

import pandas as pd

import pandas as pd

def NormalizeFeatures(df: pd.DataFrame):
    df_norm = df.copy()
    num_cols = df_norm.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_to_normalize = [col for col in num_cols if df_norm[col].nunique() > 2]
    means = df_norm[cols_to_normalize].mean()
    stds = df_norm[cols_to_normalize].std()
    df_norm[cols_to_normalize] = (df_norm[cols_to_normalize] - means) / stds

    return df_norm, means, stds

def NormalizeWithStats(df: pd.DataFrame, means: pd.Series, stds: pd.Series):
    df_norm = df.copy()
    num_cols = means.index.tolist()
    df_norm[num_cols] = (df_norm[num_cols] - means) / stds

    return df_norm


def train_pred_linear_reg(X_train, X_test, y_train, y_test, metodo="pinv", reg=None, lr=0.01, epochs=1000, l1=0, l2=0):
    modelo = LinearReg(X_train, y_train, l1=l1, l2=l2)

    if metodo == "pinv":
        modelo.train_pinv(reg=reg)
    elif metodo == "gd":
        modelo.train_gd(lr=lr, epochs=epochs, reg=reg)

    y_pred = modelo.predict(X_test)

    return {
        "modelo": modelo,
        "mse": mse(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "mae": mae(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
