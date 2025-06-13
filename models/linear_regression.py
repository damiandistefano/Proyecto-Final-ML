import numpy as np

class LinearReg:
    def __init__(self, X, y, l1=0, l2=0):
        self.X = np.column_stack((np.ones(X.shape[0]), X))
        self.y = np.array(y)
        self.W = np.zeros(self.X.shape[1])
        self.l1 = l1
        self.l2 = l2

    def train_pinv(self, reg=0):
        if reg == "l2":
            X = self.X
            I = np.eye(self.X.shape[1])
            I[0, 0] = 0  # No regularizar el término de bias
            self.W = np.linalg.inv(X.T @ X + self.l2 * I) @ X.T @ self.y
        else:
            U, S, vt = np.linalg.svd(self.X , full_matrices=False)
            S_inv = np.diag(1 / S)
            p_inv = vt.T @ S_inv @ U.T 
            self.W = p_inv @ self.y
            
    def gd(self, lr, reg):
        y_pred = self.X @ self.W
        gradient = -2 * self.X.T @ (self.y - y_pred) / len(self.y)
        if reg == "l1":
            reg_term =  self.l1 * np.sign(self.W)
        elif reg == "l2":
            reg_term = 2 * self.l2 * self.W
        else:
            reg_term = 0
        self.W -= lr * (gradient + reg_term)
        
    def train_gd(self, lr, epochs, reg=0):
        for _ in range(epochs):
            self.gd(lr, reg)
    
    def print_W(self, feature_names):
        for title, w in zip(feature_names, self.W):
            print(f"{title}: {w}")
        return self.W
    
    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return X @ self.W
    def mean_squared_error(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    
    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mean_squared_error(y_true, y_pred))

