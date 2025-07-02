import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograma_subvaluacion(df_pred):
    plt.figure(figsize=(10, 5))
    sns.histplot(df_pred["delta_%"], bins=30, kde=True, color="skyblue")
    plt.axvline(-15, color="red", linestyle="--", label="Umbral subvaluado (-15%)")
    plt.title("Distribución del Delta (%) entre precio real y predicho")
    plt.xlabel("Delta (%)")
    plt.ylabel("Cantidad de autos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def detectar_subvaluados(df_pred, marca, modelo, umbral=-15):
    """
    Devuelve los autos subvaluados para una marca y modelo dada.
    
    Parámetros:
        df_pred: DataFrame con predicciones y columnas Marca, Modelo, delta_%
        marca: string con la marca (e.g., 'Peugeot')
        modelo: string con el modelo (e.g., '2008')
        umbral: porcentaje debajo del cual se considera subvaluado (default: -15)
        
    Devuelve:
        Un DataFrame con los autos subvaluados ordenados por delta_%
    """
    filtro = (df_pred["Marca"] == marca) & (df_pred["Modelo"] == modelo)
    subvaluados = df_pred[filtro & (df_pred["delta_%"] < umbral)]
    
    if subvaluados.empty:
        print(f"No se encontraron autos subvaluados para {marca} {modelo} con delta < {umbral}%")
        return None

    subvaluados = subvaluados.sort_values(by="delta_%")
    
    cols = [
    "Marca", "Modelo", "Antigüedad", "Kilómetros", "Cilindrada", 
    "Transmisión", "Combustible", "Vendedor", 
    "precio_predicho", "Precio_usd", "delta_%"
]
    return subvaluados[cols]

def top_marca_modelo_subvaluados_relativo(df_pred, umbral=-15, top_n=5):
    # Crear una columna combinada Marca_Modelo
    df_pred["Marca_Modelo"] = df_pred["Marca"] + " " + df_pred["Modelo"]

    # Total de autos por Marca_Modelo
    total_por_mm = df_pred["Marca_Modelo"].value_counts()

    # Subvaluados
    subvaluados = df_pred[df_pred["delta_%"] < umbral]
    subvaluados_por_mm = subvaluados["Marca_Modelo"].value_counts()

    # Proporción relativa
    proporcion = (subvaluados_por_mm / total_por_mm).dropna()

    # Obtener top_n más altos
    top_relativo = proporcion.sort_values(ascending=False).head(top_n).reset_index()
    top_relativo.columns = ["Marca_Modelo", "Proporcion_subvaluados"]

    # Calcular promedios reales y predichos
    promedio_real = df_pred.groupby("Marca_Modelo")["Precio_usd"].mean()
    promedio_predicho = df_pred.groupby("Marca_Modelo")["precio_predicho"].mean()

    # Agregar precios promedios al resultado
    top_relativo["Precio_usd_promedio"] = top_relativo["Marca_Modelo"].map(promedio_real)
    top_relativo["Precio_predicho_promedio"] = top_relativo["Marca_Modelo"].map(promedio_predicho)

    return top_relativo

def plot_top_subvaluados(top_relativo):
    # Ordenar de menor a mayor para el gráfico horizontal
    top_relativo = top_relativo.sort_values("Proporcion_subvaluados", ascending=True)

    # Colores (gradiente de azul a rojo según proporción)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    colores = cmap(np.linspace(0.2, 0.8, len(top_relativo)))

    # Tamaño del gráfico
    plt.figure(figsize=(10, 6))
    
    # Barras horizontales
    bars = plt.barh(
        top_relativo["Marca_Modelo"],
        top_relativo["Proporcion_subvaluados"],
        color=colores,
        edgecolor="black"
    )

    for i, row in top_relativo.iterrows():
        texto = f"${row['Precio_usd_promedio']:.0f} → ${row['Precio_predicho_promedio']:.0f}"
        x_pos = row["Proporcion_subvaluados"] / 2  # Mitad de la barra, centrado
        plt.text(
            x_pos,
            i,
            texto,
            va="center",
            ha="center",  # alineado horizontal al centro
            fontsize=9,
            color="black"
        )


    plt.xlabel("Proporción de autos subvaluados (< -15%)")
    plt.yticks(fontsize=10)
    plt.xticks(np.linspace(0, 1, 6))
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def top_autos_subvaluados(df_pred, top_n=10):
    return df_pred.sort_values(by="delta_%").head(top_n)[[
        "Marca", "Modelo", "Antigüedad", "Kilómetros", "Cilindrada", 
         "Vendedor", "precio_predicho", "Precio_usd", "delta_%"
    ]]

