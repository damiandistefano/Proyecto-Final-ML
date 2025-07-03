import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograma_subvaluacion(df_pred):
    plt.figure(figsize=(10, 5))
    sns.histplot(df_pred["delta_%"], bins=30, kde=True, color="skyblue")
    plt.axvline(-15, color="red", linestyle="--", label="Umbral subvaluado (-15%)")
    plt.xlabel("Delta (%)")
    plt.ylabel("Cantidad de autos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def top_autos_subvaluados(df_pred, top_n=10):
    df_top = df_pred.sort_values(by="delta_%").head(top_n).copy()
    df_top.insert(0, "Ranking", range(1, top_n + 1))
    return df_top[[
        "Ranking", "Marca", "Modelo", "Antigüedad", "Cilindrada",
        "Vendedor", "precio_predicho", "Precio_usd", "delta_%"
    ]]