import numpy as np


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, X):
        X = (X - self.mean) / self.std
        return X

    def inverse_transform(self, X):
        X = X * self.std + self.mean
        return X

    def fit_transform(self, X):
        self = self.fit(X)
        return self.transform(X)
