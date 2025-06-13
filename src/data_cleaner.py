import numpy as np
import pandas as pd
class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.mean_std = []
    def preprocess(self):
        df = self.df.copy()

        # 1. Limpieza básica
        df_clean = df.drop(columns=["Unnamed: 0", "Título", "Descripción"], errors="ignore")

        # 2. Precio en USD
        usd_conversion_rate = 1185.26
        df_clean["Precio_usd"] = np.where(
            df_clean["Moneda"] == "$",
            df_clean["Precio"] / usd_conversion_rate,
            df_clean["Precio"]
        )

        # 3. Antigüedad
        df_clean["Antigüedad"] = 2025 - df_clean["Año"]

        # 4. Kilómetros a número
        df_clean["Kilómetros"] = (
            df_clean["Kilómetros"]
            .astype(str)
            .str.replace(" km", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # 5. Rellenar categóricos faltantes
        categorical_cols = ["Color", "Transmisión", "Motor", "Con cámara de retroceso"]
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("desconocido")

        # 6. One-hot encoding (manual)
        def one_hot_encode(df, column):
            unique_values = df[column].unique()
            one_hot = np.zeros((df.shape[0], len(unique_values)), dtype=float)
            value_to_index = {val: i for i, val in enumerate(unique_values)}
            for i, val in enumerate(df[column]):
                one_hot[i, value_to_index[val]] = 1
            col_names = [f"{column}_{val}" for val in unique_values]
            one_hot_df = pd.DataFrame(one_hot, columns=col_names, index=df.index)
            return one_hot_df

        columns_to_encode = [
            "Marca", "Modelo", "Color", "Tipo de combustible",
            "Transmisión", "Motor", "Tipo de carrocería",
            "Con cámara de retroceso", "Tipo de vendedor"
        ]

        encoded_parts = []
        for col in columns_to_encode:
            if col in df_clean.columns:
                encoded_parts.append(one_hot_encode(df_clean, col))

        df_encoded = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df_clean.index)

        # 7. Eliminar columnas categóricas originales si existen
        df_clean = df_clean.drop(columns=[col for col in columns_to_encode if col in df_clean.columns])
        df_clean = df_clean.drop(columns=["Versión", "Moneda"], errors="ignore")

        # 8. Agregar columnas codificadas
        df_clean = pd.concat([df_clean, df_encoded], axis=1)


        return df_clean.reset_index(drop=True)
    def normalize(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]
            mean = col.mean()
            std = col.std()
            std = 1 if std == 0 else std
            self.mean_std.append((mean, std))
            X[:, i] = (col - mean) / std
        return X
    def normalize_new_data(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]
            mean = self.mean_std[i][0]
            std = self.mean_std[i][1]
            # std = 1 if std == 0 else std
            X[:, i] = (col - mean) / std
        return X
    def get_means_std(self):
        return self.mean_std
    
 