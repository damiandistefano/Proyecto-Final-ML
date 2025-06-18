import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers=[64, 32]):
        """
        input_dim: cantidad de features de entrada
        hidden_layers: lista con cantidad de neuronas por capa oculta
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.scaler = StandardScaler()
        self.model = self._build_model()
    
    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
        
        model.add(layers.Dense(1))  # Salida: precio
        model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
        return model

    def preprocess(self, X):
        return self.scaler.transform(X)
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo con los datos dados
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    
    def predict(self, X):
        """
        Predice el precio para nuevos datos X (sin escalar)
        """
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo con los datos de prueba
        """
        X_test_scaled = self.preprocess(X_test)
        return self.model.evaluate(X_test_scaled, y_test, verbose=0)