import numpy as np
from src.metrics import mean_squared_error, r2_score


class LinearRegressionAnalytics:
    """
    Класс Аналитического решения линейной регрессии.
    При l2_coef > 0 добавляется L2 регуляризация.
    Без регуляризации решения может не существовать,
    если не существует обратной матрицы.
    """
    def __init__(self, l2_coef=0.0):
        self.l2_coef = max(0.0, l2_coef)

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


class BaseLinearRegressionGD:
    """
    Базовый класс для линейной регрессии полным градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
    """
    def __init__(self, eta, n_iter, random_state):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def _count_loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def _count_grad(self, X, y, y_pred):
        dfdw = 2 * X.T @ (y_pred - y) / X.shape[0]
        return dfdw

    def _initialize_weights(self, size):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=size)
        self.b_ = 0.0

    def fit(self, X, y):
        self._initialize_weights(X.shape[-1])
        self.losses_ = []

        for _ in range(self.n_iter):
            y_pred = X @ self.w_ + self.b_
            loss = self._count_loss(y, y_pred)
            self.losses_.append(loss)

            dfdw = self._count_grad(X, y, y_pred)
            dfdb = 2 * np.mean(y_pred - y)

            self.w_ -= self.eta * dfdw
            self.b_ -= self.eta * dfdb
        return self

    def predict(self, X):
        return X @ self.w_ + self.b_


class LinearRegressionGD(BaseLinearRegressionGD):
    """
    Класс для линейной регрессии полным градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42):
        super().__init__(eta, n_iter, random_state)


class RidgeRegressionGD(BaseLinearRegressionGD):
    """
    Класс для гребневой регрессии полным градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
        l2_coef: коэффициент L2-регуляризации
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42, l2_coef=0.0):
        super().__init__(eta, n_iter, random_state)
        self.l2_coef = max(0.0, l2_coef)

    def _count_loss(self, y_true, y_pred):
        l2_add = self.l2_coef * np.mean(self.w_ ** 2)
        return super()._count_loss(y_true, y_pred) + l2_add

    def _count_grad(self, X, y, y_pred):
        l2_add = 2 * self.l2_coef * self.w_ / X.shape[0]
        return super()._count_grad(X, y, y_pred) + l2_add
