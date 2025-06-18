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
        df_clean = df_clean.drop(columns=["Precio", "Moneda"], errors="ignore")

        # 3. Antigüedad
        df_clean["Antigüedad"] = 2025 - df_clean["Año"]
        df_clean = df_clean.drop(columns=["Año"], errors="ignore")

        # 4. Kilómetros a número
        df_clean["Kilómetros"] = (
            df_clean["Kilómetros"]
            .astype(str)
            .str.replace(" km", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

        # # 6. One-hot encoding (manual)
        # def one_hot_encode(df, column):
        #     unique_values = df[column].unique()
        #     one_hot = np.zeros((df.shape[0], len(unique_values)), dtype=float)
        #     value_to_index = {val: i for i, val in enumerate(unique_values)}
        #     for i, val in enumerate(df[column]):
        #         one_hot[i, value_to_index[val]] = 1
        #     col_names = [f"{column}_{val}" for val in unique_values]
        #     one_hot_df = pd.DataFrame(one_hot, columns=col_names, index=df.index)
        #     return one_hot_df

        # columns_to_encode = [
        #     "Marca", "Modelo", "Color", "Tipo de combustible",
        #     "Transmisión", "Motor", "Tipo de carrocería",
        #     "Con cámara de retroceso", "Tipo de vendedor"
        # ]
        # for col in columns_to_encode:
        #     print(f"{col}: {df[col].nunique()} categorías")


        # encoded_parts = []
        # for col in columns_to_encode:
        #     if col in df_clean.columns:
        #         encoded_parts.append(one_hot_encode(df_clean, col))

        # df_encoded = pd.concat(encoded_parts, axis=1) if encoded_parts else pd.DataFrame(index=df_clean.index)

        # features poco influyentes
        df_clean = df_clean.drop(columns=["Color", "Con cámara de retroceso","Versión"], errors="ignore")

        # # 7. Eliminar columnas categóricas originales si existen
        # df_clean = df_clean.drop(columns=[col for col in columns_to_encode if col in df_clean.columns])
        # df_clean = df_clean.drop(columns=["Versión", "Moneda"], errors="ignore")

        # # 8. Agregar columnas codificadas
        # df_clean = pd.concat([df_clean, df_encoded], axis=1)

        # Cilindrada en ves de motor
        df_clean['Cilindrada'] = df_clean['Motor'].str.extract(r'(\d\.\d)').astype(float)
        df_clean = df_clean.drop(columns=["Motor"], errors="ignore")
        df_clean = df_clean.dropna(subset=["Cilindrada"])

        # Agrupar las transmisiones menos comunes como "Automática"
        df_clean["Transmisión"] = df_clean["Transmisión"].replace({
            "Automática secuencial": "Automática",
            "Semiautomática": "Automática"
        })
        df_clean["Transmisión_Manual"] = (df_clean["Transmisión"] == "Manual").astype(int)
        df_clean.drop(columns=["Transmisión"], inplace=True)
        
        # Convertir "Tipo de vendedor" en binaria: 1 si es particular, 0 si es empresa (concesionaria o tienda)
        df_clean["vendedor_particular"] = df_clean["Tipo de vendedor"].apply(
            lambda x: 1 if x == "particular" else 0
        )
        df_clean.drop(columns=["Tipo de vendedor"], inplace=True)

        # ya que hay solo un valor (SUV)
        df_clean.drop(columns=["Tipo de carrocería"], inplace=True)

        # Nafta 
        otros = ["GNC", "Eléctrico", "Mild Hybrid", "Híbrido", "Híbrido/Nafta", "Nafta/GNC"]

        df_clean["Tipo de combustible agrupado"] = df_clean["Tipo de combustible"].apply(
            lambda x: x if x not in otros else "Otros"
        )
        # Ahora hacemos one-hot encoding
        df_encoded = pd.get_dummies(df_clean["Tipo de combustible agrupado"], prefix="combustible")
        df_clean = pd.concat([df_clean, df_encoded], axis=1)
        df_clean.drop(columns=["Tipo de combustible", "Tipo de combustible agrupado"], inplace=True)

        # Marcas
        alta = {"BMW", "Mercedes-Benz", "Audi", "Lexus", "Volvo", "Land Rover"}
        media = {"Toyota", "Honda", "Volkswagen", "Hyundai", "Nissan", "Chevrolet", "Ford", "Renault"}
        baja = {"Fiat", "Chery", "Peugeot", "Jetour", "JAC", "Lifan"}

        def clasificar_marca(marca):
            if marca in alta:
                return "alta"
            elif marca in media:
                return "media"
            else:
                return "baja"

        df["Gama_marca"] = df["Marca"].apply(clasificar_marca)



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
    
 