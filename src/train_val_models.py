from sklearn.model_selection import train_test_split
from models.linear_regression import LinearReg
from src.metrics import*
import pandas as pd

def split_dataset(df, target_column="Precio_usd", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_pred_linear_reg(processor, X_train, X_test, y_train, y_test, metodo="pinv", reg=None, lr=0.01, epochs=1000, l1=0, l2=0):
    modelo = LinearReg(X_train, y_train, l1=l1, l2=l2)

    if metodo == "pinv":
        modelo.train_pinv(reg=reg)
    elif metodo == "gd":
        modelo.train_gd(lr=lr, epochs=epochs, reg=reg)

    y_pred = modelo.predict(X_test)  
    y_pred_real = processor.denormalize_y(y_pred)  

    
    return {
        "modelo": modelo,
        "mse": mse(y_test, y_pred_real),
        "rmse": rmse(y_test, y_pred_real),
        "mae": mae(y_test, y_pred_real)
    }

def prepare_data(processor, dataset, target_col="Precio_usd", test_size=0.2, random_state=42):
    y = dataset[target_col]
    X = dataset.drop(columns=[target_col])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Convertimos a np.float64 para trabajar
    X_train_np = X_train.values.astype(np.float64)
    X_val_np = X_val.values.astype(np.float64)
    y_train_np = y_train.values.astype(np.float64)

    # Normalizamos con el processor
    X_train_norm = processor.normalize(X_train_np)
    X_val_norm = processor.normalize_new_data(X_val_np)
    y_train_norm = processor.normalize_y(y_train_np)

    # Convertimos de nuevo a DataFrame
    X_train_df = pd.DataFrame(X_train_norm, columns=X_train.columns)
    X_val_df = pd.DataFrame(X_val_norm, columns=X_val.columns)

    return X_train_df, X_val_df, y_train_norm, y_val  # y_train está normalizado, y_val no



def run_experiment(datasets, metodo="pinv", reg="l2", l2=0.1):
    """
    Ejecuta el pipeline completo de entrenamiento y evaluación para varios datasets.
    datasets: dict nombre -> (DataProcessor, DataFrame)
    """
    resultados = []

    for nombre, (processor, dataset) in datasets.items():
        X_train_norm, X_val_norm, y_train, y_val = prepare_data(processor, dataset)

        # Entrenamos y predecimos con el modelo
        res = train_pred_linear_reg(
            processor, X_train_norm, X_val_norm, y_train, y_val.values, ...
        )


        resultados.append({
            "nombre_dataset": nombre,
            "mse": res["mse"],
            "rmse": res["rmse"],
            "mae": res["mae"]
        })

        print(f"[{nombre}] MSE: {res['mse']:.2f} | RMSE: {res['rmse']:.2f} | MAE: {res['mae']:.2f}")

    return resultados
