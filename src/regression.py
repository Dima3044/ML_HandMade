import numpy as np


class LinearRegressionAnalytics:
    def __init__(self, l2_coef=0.0):
        self.l2_coef = l2_coef

    def fit(self, X, y):
        b_col = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.hstack([b_col, X])
        if self.l2_coef > 0:
            I = np.eye(X.shape[-1])
            I[0][0] = 0
            w = np.linalg.inv(X.T @ X + I * self.l2_coef) @ X.T @ y
        else:
            w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.w_ = w[1:]
        self.b_ = w[0]
        return self

    def predict(self, X):
        return X @ self.w_ + self.b_


class LinearRegressionGD:
    def __init__(self, eta=0.1, n_iter=50, l2_coef=0.0, random_state=42):
        self.eta = eta
        self.n_iter = n_iter
        self.l2_coef = l2_coef
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[-1])
        self.b_ = 0.0
        self.losses_ = []
        N = X.shape[0]

        for _ in range(self.n_iter):
            y_pred = X @ self.w_ + self.b_
            loss = np.mean((y-y_pred)**2)
            if self.l2_coef > 0:
                loss += self.l2_coef * np.mean(self.w_ ** 2)
            self.losses_.append(loss)

            dfdw = 2 * X.T @ (y_pred - y) / N
            if self.l2_coef > 0:
                dfdw += 2 * self.l2_coef * self.w_ / N
            dfdb = 2 * np.mean(y_pred - y)

            self.w_ -= self.eta * dfdw
            self.b_ -= self.eta * dfdb
        return self

    def predict(self, X):
        return X @ self.w_ + self.b_
