from matplotlib import pyplot as plt
import numpy as np

def plot_val_test_rmse(results):
    """
    Grafica barras comparativas de RMSE de validación y test.
    El color de cada barra se asigna según cuál tiene menor error.
    """

    # Nombres acortados con saltos de línea para ahorrar espacio
    labels = [
        "Feature engineering\ny agrupando marcas",
        "One-hot agrupando\nsolo marcas",
        "One-hot con\nagrupamiento",
        "One-hot sin\nagrupamiento",
        "Sin marcas ni modelos,\ncon agrupamiento",
        "Sin marcas ni modelos,\nsin agrupamiento"
    ]
    
    val_scores = [r['val_rmse'] for r in results]
    test_scores = [r['test_rmse'] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_val = ax.bar(x - width/2, val_scores, width, label='Validación', color='#4C72B0')
    bars_test = ax.bar(x + width/2, test_scores, width, label='Test', color='#55A868')

    # Mostrar el valor de RMSE sobre cada barra
    for bar in bars_val:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos más arriba
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars_test:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=9, ha='center')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()