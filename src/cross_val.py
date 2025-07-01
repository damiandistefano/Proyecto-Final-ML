import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from itertools import product


def parse_max_features(value, n_features):
    if isinstance(value, str):
        return value  # "sqrt", "log2" ya son válidos para sklearn
    return min(value, n_features)  # asegurar que no supere la cantidad de features


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cross_validate_rf(df, target_col, param_grid, k=5, metric_fn=None, random_state=42):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # shuffle
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    if metric_fn is None:
        metric_fn = rmse

    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    best_score = float('inf')  # porque RMSE
    best_params = None
    all_scores = []

    for combo in combinations:
        params = dict(zip(keys, combo))
        fold_scores = []

        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            max_feats = parse_max_features(params.get("max_features"), X.shape[1])

            model = RandomForestRegressor(
                n_estimators=params.get("n_trees", 10),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                max_features=max_feats,
                random_state=random_state,
                n_jobs=-1  # usa todos los núcleos disponibles
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = metric_fn(y_val, y_pred)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        all_scores.append((params, avg_score))

        print(f"{params} --> RMSE promedio: {avg_score:.4f}")

        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    print("\n✅ Mejores hiperparámetros:", best_params)
    print(f"🏆 Score promedio (RMSE): {best_score:.4f}")
    return best_params, all_scores


from sklearn.model_selection import train_test_split


def tune_and_test_rf(
    df,
    target_col,
    param_grid,
    top_n=3,
    test_size=0.2,
    random_state=42,
    metric_fn=None
):
    """
    Realiza la búsqueda de hiperparámetros con validación cruzada en train+val,
    luego evalúa las top_n configuraciones en test para elegir la mejor.

    Retorna un diccionario con resultados y mejores parámetros.
    """
   

    # Separar en trainval y test
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    # Buscar mejores params con validación cruzada
    best_params, scores = cross_validate_rf(
        df_trainval,
        target_col=target_col,
        param_grid=param_grid,
        metric_fn=metric_fn,
        random_state=random_state
    )

    # Ordenar scores y tomar top_n configs
    scores_sorted = sorted(scores, key=lambda x: x[1])
    top_params = [params for params, _ in scores_sorted[:top_n]]

    X_trainval = df_trainval.drop(columns=[target_col])
    y_trainval = df_trainval[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    best_test_rmse = float('inf')
    best_final_params = None

    for params in top_params:
        model = RandomForestRegressor(
            n_estimators=params["n_trees"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_trainval, y_trainval)
        y_pred_test = model.predict(X_test)
        rmse_test = metric_fn(y_test, y_pred_test) if metric_fn else rmse(y_test, y_pred_test)
        print(f"Test RMSE con params {params}: {rmse_test:.4f}")

        if rmse_test < best_test_rmse:
            best_test_rmse = rmse_test
            best_final_params = params

    return {
        "best_params_val": best_params,
        "val_rmse": min(score for _, score in scores),
        "best_params_test": best_final_params,
        "test_rmse": best_test_rmse
        }

def analyze_datasets_performance(datasets_explorations, param_grid, resultados):
    for nombre, dataset in datasets_explorations:
        print(f"\nEvaluando dataset: {nombre}")
        res = tune_and_test_rf(dataset, target_col="Precio_usd", param_grid=param_grid, top_n=3)
        print(res)
        resultados.append({
    "dataset": nombre,
    "best_params_val": res["best_params_val"],
    "val_rmse": res["val_rmse"],
    "best_params_test": res["best_params_test"],
    "test_rmse": res["test_rmse"]
})