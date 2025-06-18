import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, df, config=None):
        self.df = df
        self.mean_std = []
        self.config = config or {
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
            "group_combustible": True
        }

    def preprocess(self):
        df = self.df.copy()

        # 1. Limpieza básica
        if self.config["clean_columns"]:
            df = df.drop(columns=["Unnamed: 0", "Título", "Descripción"], errors="ignore")

        # 2. Arreglar marcas mal escritas
        if self.config["fix_brand_typos"]:
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

        # 6. One-hot de Marca y Modelo
        if self.config["one_hot_encode"]:
            for col in ["Marca", "Modelo"]:
                if col in df.columns:
                    one_hot = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, one_hot], axis=1)
                    df = df.drop(columns=[col])

        # 7. Eliminar columnas de poca información
        if self.config["drop_low_info"]:
            df = df.drop(columns=["Color", "Con cámara de retroceso", "Versión", "Tipo de carrocería"], errors="ignore")

        # 8. Extraer cilindrada
        if self.config["parse_motor"]:
            df["Cilindrada"] = df["Motor"].str.extract(r'(\d\.\d)').astype(float)
            df.drop(columns=["Motor"], inplace=True)
            df.dropna(subset=["Cilindrada"], inplace=True)

        # 9. Agrupar transmisiones y binarizar
        if self.config["group_transmission"]:
            df["Transmisión"] = df["Transmisión"].replace({
                "Automática secuencial": "Automática",
                "Semiautomática": "Automática"
            })
            df["Transmisión_Manual"] = (df["Transmisión"] == "Manual").astype(int)
            df.drop(columns=["Transmisión"], inplace=True)

        # 10. Vendedor particular
        if self.config["encode_vendedor"]:
            df["vendedor_particular"] = (df["Tipo de vendedor"] == "particular").astype(int)
            df.drop(columns=["Tipo de vendedor"], inplace=True)

        # 11. Agrupar combustibles poco comunes
        if self.config["group_combustible"]:
            otros = ["GNC", "Eléctrico", "Mild Hybrid", "Híbrido", "Híbrido/Nafta", "Nafta/GNC"]
            df["Tipo de combustible agrupado"] = df["Tipo de combustible"].apply(
                lambda x: x if x not in otros else "Otros"
            )
            dummies = pd.get_dummies(df["Tipo de combustible agrupado"], prefix="combustible")
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=["Tipo de combustible", "Tipo de combustible agrupado"], errors="ignore")

        return df.reset_index(drop=True)


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
    
 