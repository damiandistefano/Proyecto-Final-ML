import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Exploración inicial

def exploratory_analysis(df, target_col):
    
    print(df.info())

    print("\n--- Resumen Estadístico ---")
    print(df.describe())

    print("\n--- Valores Faltantes ---")
    print(df.isnull().sum())

    print("\n--- Distribución de la Variable Objetivo ---")
    print(df[target_col].value_counts(normalize=True))


def box_plot_features(df, target_col='Precio_usd'):
    
    # Seleccionar columnas numéricas excluyendo target y variables dummy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Estilo de Seaborn
    sns.set(style="whitegrid", palette="pastel")

    # Crear el gráfico
    plt.figure(figsize=(16, 6))
    ax = sns.boxplot(data=df[numeric_cols], width=0.5, fliersize=3)

    # Etiquetas y ajustes
    ax.set_title("Distribución de Variables Numéricas", fontsize=16, weight='bold')
    ax.set_xlabel("Variables", fontsize=12)
    ax.set_ylabel("Valor", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_price_vs_nafta_type(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Tipo de combustible', y='Precio_usd', data=df)
    plt.xticks(rotation=45)
    plt.title('Distribución de Precio según Tipo de Nafta')
    plt.show()

