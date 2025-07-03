import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, df, config=None):
        self.df = df
        self.mean_std = []
        self.y_mean = None
        self.y_std = None
        self.one_hot_categories = {}
        

        default_config = {
            "clean_columns": True,
            "fix_brand_typos": True,
            "convert_price": True,
            "calc_antiguedad": True,
            "convert_km": True,
            "one_hot_encode": True,
            "drop_low_info": True,
            "parse_motor": True,
            "group_transmission": True,
            "encode_vendedor": True,
            "group_combustible": True,
            "add_precio_por_km": False,
            "add_antiguedad_squared": False,
            "add_cilindrada_times_km": False,
            "add_frecuencia_features": False,
            "outlaier_group": True,
            "limpieza_de_outliers": True
        }

        self.config = default_config.copy()
        if config:
            self.config.update(config)

    def preprocess_global(self):
        df = self.df.copy()

        if self.config.get("clean_columns", True):
            df = df.drop(columns=["Unnamed: 0", "Título", "Descripción"], errors="ignore")

        if self.config.get("fix_brand_typos", True) and "Marca" in df.columns:
            df["Marca"] = df["Marca"].replace({
                "Hiunday": "Hyundai",
                "hiunday": "Hyundai",
                "Rrenault": "Renault",
                "Jetur": "Jetour",
                "Vol": "Volvo"
            })

        if self.config.get("convert_price", True):
            usd_conversion_rate = 1185.26
            df["Precio_usd"] = np.where(df["Moneda"] == "$", df["Precio"] / usd_conversion_rate, df["Precio"])
            df = df.drop(columns=["Precio", "Moneda"], errors="ignore")

        if self.config.get("calc_antiguedad", True):
            df["Antigüedad"] = 2025 - df["Año"]
            df = df.drop(columns=["Año"], errors="ignore")

        if self.config.get("convert_km", True):
            df["Kilómetros"] = (
                df["Kilómetros"]
                .astype(str)
                .str.replace(" km", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )


        if self.config.get("drop_low_info", True):
            df = df.drop(columns=["Color", "Con cámara de retroceso", "Versión", "Tipo de carrocería"], errors="ignore")

        if self.config.get("parse_motor", True):
            df["Cilindrada"] = df["Motor"].str.extract(r'(\d\.\d)').astype(float)
            df.drop(columns=["Motor"], inplace=True)
            df.dropna(subset=["Cilindrada"], inplace=True)

        if self.config.get("group_transmission", True):
            df["Transmisión"] = df["Transmisión"].replace({
                "Automática secuencial": "Automática",
                "Semiautomática": "Automática"
            })
            df["Transmisión_Manual"] = (df["Transmisión"] == "Manual").astype(int)
            df.drop(columns=["Transmisión"], inplace=True)
        elif self.config.get("group_transmission") is False:
            dummies = pd.get_dummies(df["Transmisión"], prefix="transmision")
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=["Transmisión"], inplace=True)

        if self.config.get("encode_vendedor", True):
            df["vendedor_particular"] = (df["Tipo de vendedor"] == "particular").astype(int)
            df.drop(columns=["Tipo de vendedor"], inplace=True)

        if "group_combustible" in self.config:
            if self.config["group_combustible"]:
                otros = ["GNC", "Eléctrico", "Mild Hybrid", "Híbrido", "Híbrido/Nafta", "Nafta/GNC", "Híbrido/Diesel"]
                df["Tipo de combustible agrupado"] = df["Tipo de combustible"].apply(
                    lambda x: x if x not in otros else "Otros"
                )
                dummies = pd.get_dummies(df["Tipo de combustible agrupado"], prefix="combustible")
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=["Tipo de combustible", "Tipo de combustible agrupado"], errors="ignore")
            else:
                dummies = pd.get_dummies(df["Tipo de combustible"], prefix="combustible")
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=["Tipo de combustible"], errors="ignore")

        if self.config.get("add_precio_por_km", False):
            df["precio_por_km"] = df["Precio_usd"] / (df["Kilómetros"] + 1)

        if self.config.get("add_antiguedad_squared", False):
            df["antiguedad_squared"] = df["Antigüedad"] ** 2

        if self.config.get("add_cilindrada_times_km", False):
            if "Cilindrada" in df.columns and "Kilómetros" in df.columns:
                df["cilindrada_x_km"] = df["Cilindrada"] * df["Kilómetros"]

        if self.config.get("add_frecuencia_features", False):
            if "Marca" in self.df.columns:
                freq_marca = self.df["Marca"].value_counts(normalize=True)
                df["frecuencia_marca"] = self.df["Marca"].map(freq_marca)
            if "Modelo" in self.df.columns:
                freq_modelo = self.df["Modelo"].value_counts(normalize=True)
                df["frecuencia_modelo"] = self.df["Modelo"].map(freq_modelo)

        self.df = df
        return df.reset_index(drop=True)

    def preprocess_split(self):
        df = self.df.copy()
        if self.config.get("outlaier_group", False):
            umbral = 0.01
            for col in ["Marca"]:
                if col in df.columns:
                    freq = df[col].value_counts(normalize=True)
                    frecuentes = freq[freq >= umbral].index
                    df[col] = df[col].apply(lambda x: x if x in frecuentes else f"{col}_Otros")

        if self.config.get("limpieza_de_outliers", True):
            df_clean = df.copy()
            df_clean = df_clean[df_clean['Antigüedad'] >= 0]
            df_clean = df_clean[df_clean['Kilómetros'] >= 0]

            for col, factor in [("Kilómetros", 1.5), ("Antigüedad", 2.5)]:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lim_inf = q1 - factor * iqr
                lim_sup = q3 + factor * iqr
                df_clean = df_clean[(df_clean[col] >= lim_inf) & (df_clean[col] <= lim_sup)]

            df_clean = df_clean.drop_duplicates()
            df = df_clean

        if self.config.get("one_hot_encode", True):
            for col in ["Marca", "Modelo"]:
                if col in df.columns:
                    if col not in self.one_hot_categories:
                        self.one_hot_categories[col] = sorted(df[col].dropna().unique())

                    categories = self.one_hot_categories[col]
                    one_hot = pd.get_dummies(df[col], prefix=col)

                    for cat in categories:
                        col_name = f"{col}_{cat}"
                        if col_name not in one_hot.columns:
                            one_hot[col_name] = 0

                    expected_cols = [f"{col}_{c}" for c in categories]
                    one_hot = one_hot[expected_cols]

                    df = df.drop(columns=[col])
                    df = pd.concat([df, one_hot], axis=1)

        df = df.astype(float)
        return df.reset_index(drop=True)

    def preprocess_new_data(self, new_df):
        temp = DataProcessor(new_df, config={
            "clean_columns": True,
            "fix_brand_typos": True,
            "convert_price": False,
            "calc_antiguedad": True,
            "convert_km": True,
            "one_hot_encode": True,
            "drop_low_info": True,
            "parse_motor": True,
            "group_transmission": True,
            "encode_vendedor": True,
            "group_combustible": True,
            "add_precio_por_km": False,
            "add_antiguedad_squared": False,
            "add_cilindrada_times_km": False,
            "add_frecuencia_features": False,
            "outlaier_group": False,
            "limpieza_de_outliers": False
        })
        temp.one_hot_categories = self.one_hot_categories
        temp_df = temp.preprocess_split() # usar las mismas categorías
        return temp_df

    def normalize(self, X):
        self.mean_std = []  # Reiniciar por si se reutiliza el objeto

        for i in range(X.shape[1]):
            col = X[:, i]
            unique_vals = np.unique(col)

            # Detectar si es one-hot
            if set(unique_vals).issubset({0, 1}):
                self.mean_std.append((0, 1))  # No normalizamos, guardamos como identidad
                continue

            mean = col.mean()
            std = col.std()
            std = 1 if std == 0 else std  # Evitar división por cero
            self.mean_std.append((mean, std))
            X[:, i] = (col - mean) / std

        return X

    def normalize_new_data(self, X):
        for i in range(X.shape[1]):
            mean, std = self.mean_std[i]

            if std == 1 and mean == 0:
                continue  # Era columna one-hot, no normalizar

            col = X[:, i]
            X[:, i] = (col - mean) / std

        return X
    
    def normalize_y(self, y):
        """
        Normaliza un array 1D de y (target) y guarda su media y std.
        """
        self.y_mean = y.mean()
        self.y_std = y.std() if y.std() != 0 else 1.0
        return (y - self.y_mean) / self.y_std
    
    def denormalize_y(self, y_norm):
        """
        Denormaliza un array 1D de y (target) usando la media y std guardadas.
        """
        return y_norm * self.y_std + self.y_mean

        
    def get_means_std(self):
        return self.mean_std
    
 