import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, df, config=None):
        self.df = df
        self.mean_std = []
        self.y_mean = None
        self.y_std = None
        

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

    def preprocess(self):
        df = self.df.copy()

        # 1. Limpieza básica
        if self.config["clean_columns"]:
            df = df.drop(columns=["Unnamed: 0", "Título", "Descripción"], errors="ignore")

        # 2. Arreglar marcas mal escritas
        if self.config["fix_brand_typos"] and "Marca" in df.columns:
            df["Marca"] = df["Marca"].replace({
                "Hiunday": "Hyundai",
                "hiunday": "Hyundai",
                "Rrenault": "Renault",
                "Jetur": "Jetour",
                "Vol": "Volvo"
            })

        # 3. Precio a USD
        if self.config["convert_price"]:
            usd_conversion_rate = 1185.26
            df["Precio_usd"] = np.where(df["Moneda"] == "$", df["Precio"] / usd_conversion_rate, df["Precio"])
            df = df.drop(columns=["Precio", "Moneda"], errors="ignore")

        # 4. Antigüedad
        if self.config["calc_antiguedad"]:
            df["Antigüedad"] = 2025 - df["Año"]
            df = df.drop(columns=["Año"], errors="ignore")

        # 5. Convertir Kilómetros
        if self.config["convert_km"]:
            df["Kilómetros"] = (
                df["Kilómetros"]
                .astype(str)
                .str.replace(" km", "", regex=False)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

        # Agrupación de outliers en categorías poco frecuentes
        if self.config["outlaier_group"]:
            umbral = 0.01
            for col in ["Marca"]:
                if col in df.columns:
                    freq = df[col].value_counts(normalize=True)
                    frecuentes = freq[freq >= umbral].index
                    df[col] = df[col].apply(lambda x: x if x in frecuentes else f"{col}_Otros")

        # 6. Limpieza de outliers
        if self.config.get("limpieza_de_outliers", True):
            df_clean = df.copy()
            inicial = len(df_clean)

            df_clean = df_clean[df_clean['Antigüedad'] >= 0]
            df_clean = df_clean[df_clean['Kilómetros'] >= 0]

            q1 = df_clean['Kilómetros'].quantile(0.25)
            q3 = df_clean['Kilómetros'].quantile(0.75)
            iqr = q3 - q1
            lim_inf = q1 - 1.5 * iqr
            lim_sup = q3 + 1.5 * iqr
            df_clean = df_clean[(df_clean['Kilómetros'] >= lim_inf) & (df_clean['Kilómetros'] <= lim_sup)]

            q1 = df_clean['Antigüedad'].quantile(0.25)
            q3 = df_clean['Antigüedad'].quantile(0.75)
            iqr = q3 - q1
            lim_inf = q1 - 2.5 * iqr
            lim_sup = q3 + 2.5 * iqr
            df_clean = df_clean[(df_clean['Antigüedad'] >= lim_inf) & (df_clean['Antigüedad'] <= lim_sup)]

            df_clean = df_clean.drop_duplicates()

            print(f"Filas iniciales: {inicial} → después del filtrado: {len(df_clean)}")
            df = df_clean

        # 7. One-hot de Marca y Modelo (modificado)
        if self.config["one_hot_encode"]:
            for col in ["Marca", "Modelo"]:
                if col in df.columns:
                    # Guardar categorías únicas al entrenar
                    if col not in self.one_hot_categories:
                        self.one_hot_categories[col] = sorted(df[col].dropna().unique())

                    categories = self.one_hot_categories[col]

                    one_hot = pd.get_dummies(df[col], prefix=col)

                    # Asegurar que todas las columnas originales estén
                    for cat in categories:
                        col_name = f"{col}_{cat}"
                        if col_name not in one_hot.columns:
                            one_hot[col_name] = 0

                    expected_cols = [f"{col}_{c}" for c in categories]
                    one_hot = one_hot[expected_cols]

                    df = df.drop(columns=[col])
                    df = pd.concat([df, one_hot], axis=1)

        # 8. Eliminar columnas de poca información
        if self.config["drop_low_info"]:
            df = df.drop(columns=["Color", "Con cámara de retroceso", "Versión", "Tipo de carrocería"], errors="ignore")

        # 9. Extraer cilindrada
        if self.config["parse_motor"]:
            df["Cilindrada"] = df["Motor"].str.extract(r'(\d\.\d)').astype(float)
            df.drop(columns=["Motor"], inplace=True)
            df.dropna(subset=["Cilindrada"], inplace=True)

        # 10. Agrupar transmisiones
        if self.config["group_transmission"]:
            df["Transmisión"] = df["Transmisión"].replace({
                "Automática secuencial": "Automática",
                "Semiautomática": "Automática"
            })
            df["Transmisión_Manual"] = (df["Transmisión"] == "Manual").astype(int)
            df.drop(columns=["Transmisión"], inplace=True)
        elif self.config["group_transmission"] is False:
            dummies = pd.get_dummies(df["Transmisión"], prefix="transmision")
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=["Transmisión"], inplace=True)

        # 11. Vendedor particular
        if self.config["encode_vendedor"]:
            df["vendedor_particular"] = (df["Tipo de vendedor"] == "particular").astype(int)
            df.drop(columns=["Tipo de vendedor"], inplace=True)

        # 12. Combustible
        if self.config["group_combustible"]:
            otros = ["GNC", "Eléctrico", "Mild Hybrid", "Híbrido", "Híbrido/Nafta", "Nafta/GNC"]
            df["Tipo de combustible agrupado"] = df["Tipo de combustible"].apply(
                lambda x: x if x not in otros else "Otros"
            )
            dummies = pd.get_dummies(df["Tipo de combustible agrupado"], prefix="combustible")
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=["Tipo de combustible", "Tipo de combustible agrupado"], errors="ignore")
        elif self.config["group_combustible"] is False:
            dummies = pd.get_dummies(df["Tipo de combustible"], prefix="combustible")
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=["Tipo de combustible"], errors="ignore")

        # 13. Precio por kilómetro
        if self.config.get("add_precio_por_km", False) and "Precio_usd" in df.columns and "Kilómetros" in df.columns:
            df["precio_por_km"] = df["Precio_usd"] / (df["Kilómetros"] + 1)

        # 14. Antigüedad al cuadrado
        if self.config.get("add_antiguedad_squared", False) and "Antigüedad" in df.columns:
            df["antiguedad_squared"] = df["Antigüedad"] ** 2

        # 15. Interacciones entre variables
        if self.config.get("add_cilindrada_times_km", False):
            if "Cilindrada" in df.columns and "Kilómetros" in df.columns:
                df["cilindrada_x_km"] = df["Cilindrada"] * df["Kilómetros"]

        # 16. Rareza de marca y modelo
        if self.config.get("add_frecuencia_features", False):
            if "Marca" in self.df.columns:
                freq_marca = self.df["Marca"].value_counts(normalize=True)
                df["frecuencia_marca"] = self.df["Marca"].map(freq_marca)

            if "Modelo" in self.df.columns:
                freq_modelo = self.df["Modelo"].value_counts(normalize=True)
                df["frecuencia_modelo"] = self.df["Modelo"].map(freq_modelo)

        df = df.astype(float)
        return df.reset_index(drop=True)

    

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
    
 


