import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import itertools
import numpy as np
from src.data_cleaner import DataProcessor
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers=[64, 32], optimizer_name='adam', learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model = self.build_model()
    
    def build_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))

        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))

        model.add(layers.Dense(1))

        optimizer = self.build_optimizer()
        model.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])
        return model

    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2, early_stopping=True, patience=20, validation_data=None):
        callbacks = []
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
        # lo tengo que hacer asi me deja hacer monitor val loss
        if validation_data is not None:
            self.model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=callbacks,
                        verbose=1,
                        shuffle=True)
        else:
            self.model.fit(X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=callbacks,
                        verbose=1, 
                        shuffle=True)
    
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test, verbose=0)


# def cross_validate_nn(X, y, param_grid, epochs=50, batch_size=32, k=3):
#     results = []

#     indices = np.arange(len(X))
#     X = X.iloc[indices]
#     y = y.iloc[indices]

#     # Creamos los índices para los folds
#     fold_sizes = np.full(k, len(X) // k)
#     fold_sizes[:len(X) % k] += 1
#     current = 0
#     folds = []
#     for fold_size in fold_sizes:
#         start, stop = current, current + fold_size
#         folds.append((start, stop))
#         current = stop

#     all_params = list(itertools.product(
#         param_grid['hidden_layers'],
#         param_grid['optimizer'],
#         param_grid['learning_rate']
#     ))

#     for hidden_layers, optimizer, lr in all_params:
#         print(f"Evaluando: layers={hidden_layers}, optimizer={optimizer}, lr={lr}")
#         fold_mse = []

#         for i in range(k):
#             val_start, val_end = folds[i]
#             X_val_raw = X.iloc[val_start:val_end]
#             y_val = y.iloc[val_start:val_end]

#             X_train_raw = pd.concat([X.iloc[:val_start], X.iloc[val_end:]])
#             y_train = pd.concat([y.iloc[:val_start], y.iloc[val_end:]])

#             dp = DataProcessor(df=X_train_raw)
#             X_train = dp.preprocess_split().to_numpy()
#             X_val = dp.preprocess_new_data(X_val_raw).to_numpy()

#             X_train = dp.normalize(X_train.copy())
#             X_val = dp.normalize_new_data(X_val.copy())

#             input_dim = X_train.shape[1]
#             nn = NeuralNetwork(input_dim=input_dim,
#                                hidden_layers=hidden_layers,
#                                optimizer_name=optimizer,
#                                learning_rate=lr)
#             nn.fit(X_train, y_train,
#                    epochs=epochs,
#                    batch_size=batch_size,
#                    validation_data=(X_val, y_val))

#             y_pred = nn.predict(X_val)
#             mse = mean_squared_error(y_val, y_pred)
#             fold_mse.append(mse)

#         avg_mse = np.mean(fold_mse)
#         results.append({
#             'hidden_layers': hidden_layers,
#             'optimizer': optimizer,
#             'learning_rate': lr,
#             'avg_val_mse': avg_mse
#         })

#     return results

def cross_validate_nn(df, param_grid, epochs=50, batch_size=32, k=3):
    results = []

    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

    # Creamos los índices para los folds
    fold_sizes = np.full(k, len(df) // k)
    fold_sizes[:len(df) % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((start, stop))
        current = stop

    all_params = list(itertools.product(
        param_grid['hidden_layers'],
        param_grid['optimizer'],
        param_grid['learning_rate']
    ))

    for hidden_layers, optimizer, lr in all_params:
        print(f"Evaluando: layers={hidden_layers}, optimizer={optimizer}, lr={lr}")
        fold_mse = []

        for i in range(k):
            val_start, val_end = folds[i]
            df_val_raw = df.iloc[val_start:val_end]
            df_train_raw = pd.concat([df.iloc[:val_start], df.iloc[val_end:]])

            dp = DataProcessor(df=df_train_raw)
            df_train_processed = dp.preprocess_split()  # Solo features
            df_val_processed = dp.preprocess_new_data(df_val_raw)

            # Separar X e y (por ejemplo "Precio_usd" como target)
            y_train = df_train_processed["Precio_usd"].to_numpy()
            X_train = df_train_processed.drop(columns=["Precio_usd"]).to_numpy()

            y_val = df_val_processed["Precio_usd"].to_numpy()
            X_val = df_val_processed.drop(columns=["Precio_usd"]).to_numpy()

            # Normalizar
            X_train = dp.normalize(X_train.copy())
            X_val = dp.normalize_new_data(X_val.copy())

            input_dim = X_train.shape[1]
            nn = NeuralNetwork(input_dim=input_dim,
                               hidden_layers=hidden_layers,
                               optimizer_name=optimizer,
                               learning_rate=lr)
            nn.fit(X_train, y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(X_val, y_val))

            y_pred = nn.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            fold_mse.append(mse)

        avg_mse = np.mean(fold_mse)
        results.append({
            'hidden_layers': hidden_layers,
            'optimizer': optimizer,
            'learning_rate': lr,
            'avg_val_mse': avg_mse
        })

    return results

