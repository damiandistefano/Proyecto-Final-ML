import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import itertools
import numpy as np
from src.data_cleaner import DataProcessor

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers=[64, 32], optimizer_name='adam', learning_rate=0.001):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))

        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))

        model.add(layers.Dense(1))

        optimizer = self._build_optimizer()
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


# def cross_validate_nn(X, y, param_grid, input_dim, epochs=50, batch_size=32, k=3):
#     results = []
#     # columns = X.columns
#     X_np = X.to_numpy()
#     y_np = y.to_numpy()

#     # Generar todas las combinaciones posibles de hiperparámetros
#     all_params = list(itertools.product(
#         param_grid['hidden_layers'],
#         param_grid['optimizer'],
#         param_grid['learning_rate']
#     ))

#     for hidden_layers, optimizer, lr in all_params:
#         print(f"Evaluando: layers={hidden_layers}, optimizer={optimizer}, lr={lr}")
#         fold_mse = []

#         kf = KFold(n_splits=k, shuffle=True, random_state=42)

#         for train_index, val_index in kf.split(X):
#             X_train_raw, X_val_raw = X_np[train_index], X_np[val_index]
#             y_train, y_val = y_np[train_index], y_np[val_index]

#             dp = DataProcessor(df=None)  
#             X_train = dp.normalize(X_train_raw.copy())
#             X_val = dp.normalize_new_data(X_val_raw.copy())

#             # Crear y entrenar modelo
#             nn = NeuralNetwork(input_dim=input_dim,
#                                hidden_layers=hidden_layers,
#                                optimizer_name=optimizer,
#                                learning_rate=lr)
#             nn.fit(X_train, y_train,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 validation_data=(X_val, y_val))

#             # Evaluar
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
def cross_validate_nn(X, y, param_grid, input_dim, epochs=50, batch_size=32, k=3):
    results = []
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    indices = np.arange(len(X_np))
    X_np = X_np[indices]
    y_np = y_np[indices]

    # Creamos los índices para los folds
    fold_sizes = np.full(k, len(X_np) // k)
    fold_sizes[:len(X_np) % k] += 1  # Distribuir sobrantes
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append((start, stop))
        current = stop

    # Generar todas las combinaciones de hiperparámetros
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
            X_val_raw = X_np[val_start:val_end]
            y_val = y_np[val_start:val_end]

            X_train_raw = np.concatenate([X_np[:val_start], X_np[val_end:]])
            y_train = np.concatenate([y_np[:val_start], y_np[val_end:]])

            dp = DataProcessor(df=None)
            X_train = dp.normalize(X_train_raw.copy())
            X_val = dp.normalize_new_data(X_val_raw.copy())

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