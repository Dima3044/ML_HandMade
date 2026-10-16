import numpy as np
from src.metrics import mean_squared_error, r2_score


class LinearRegressionAnalytics:
    """
    Класс Аналитического решения линейной регрессии.
    При alpha > 0 добавляется L2 регуляризация.
    Без регуляризации решения может не существовать,
    если не существует обратной матрицы.
    """
    def __init__(self, alpha=0.0):
        self.alpha = max(0.0, alpha)

    def fit(self, X, y):
        b_col = np.ones(X.shape[0]).reshape(-1, 1)
        X = np.hstack([b_col, X])
        if self.alpha > 0:
            I = np.eye(X.shape[-1])
            I[0][0] = 0
            w = np.linalg.inv(X.T @ X + I * self.alpha) @ X.T @ y
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

    def _grad_step(self, dfdw, dfdb):
        self.w_ -= self.eta * dfdw
        self.b_ -= self.eta * dfdb

    def fit(self, X, y, X_test=None, y_test=None):
        self._initialize_weights(X.shape[-1])
        self.losses_ = []
        self.scores_ = []
        if X_test is not None and y_test is not None:
            self.test_losses_ = []
            self.test_scores_ = []

        for _ in range(self.n_iter):
            y_pred = X @ self.w_ + self.b_
            loss = self._count_loss(y, y_pred)
            score = r2_score(y, y_pred)
            self.losses_.append(loss)
            self.scores_.append(score)
            if X_test is not None and y_test is not None:
                y_test_pred = X_test @ self.w_ + self.b_
                test_loss = self._count_loss(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                self.test_losses_.append(test_loss)
                self.test_scores_.append(test_r2)
            dfdw = self._count_grad(X, y, y_pred)
            dfdb = 2 * np.mean(y_pred - y)
            self._grad_step(dfdw, dfdb)
        return self
        

    def predict(self, X):
        return X @ self.w_ + self.b_


class LinearRegression(BaseLinearRegressionGD):
    """
    Класс для линейной регрессии полным градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42):
        super().__init__(eta, n_iter, random_state)


class Ridge(BaseLinearRegressionGD):
    """
    Класс для гребневой (L2) регрессии полным градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
        alpha: коэффициент L2-регуляризации
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42, alpha=0.0):
        super().__init__(eta, n_iter, random_state)
        self.alpha = max(0.0, alpha)

    def _count_loss(self, y_true, y_pred):
        l2_add = self.alpha * np.sum(self.w_ ** 2)
        return super()._count_loss(y_true, y_pred) + l2_add

    def _count_grad(self, X, y, y_pred):
        l2_add = 2 * self.alpha * self.w_
        return super()._count_grad(X, y, y_pred) + l2_add


class Lasso(BaseLinearRegressionGD):
    """
    Класс для лассо (L1) регрессии полным проксимальным
    градиентным спуском.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
        alpha: коэффициент L1-регуляризации
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42, alpha=0.0):
        super().__init__(eta, n_iter, random_state)
        self.alpha = max(0, alpha)

    def _soft_threshold(self, z):
        threshold = self.eta * self.alpha
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    def _count_loss(self, y_true, y_pred):
        l1_add = self.alpha * np.sum(np.abs(self.w_))
        return super()._count_loss(y_true, y_pred) + l1_add

    def _grad_step(self, dfdw, dfdb):
        self.w_ = self._soft_threshold(self.w_ - self.eta * dfdw)
        self.b_ -= self.eta * dfdb


class ElasticNet(BaseLinearRegressionGD):
    """
    Класс для Эластичной сети полным проксимальным
    градиентным спуском. Сочетает L1 и L2 регуляризации.
    Args:
        eta: шаг обучения
        n_iter: число итераций
        random_state: для воспроизводимости
        alpha: общий коэффициент регуляризации
        l1_ratio: доля L1 регуляризации от alpha
    """
    def __init__(self, eta=0.1, n_iter=50, random_state=42, alpha=0.0,
                 l1_ratio=0.0):
        super().__init__(eta, n_iter, random_state)
        self.alpha = max(0, alpha)
        self.l1_ratio = min(max(0, l1_ratio), 1.0)

    def _soft_threshold(self, z):
        threshold = self.eta * self.alpha * self.l1_ratio
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    def _count_loss(self, y_true, y_pred):
        l1_add = self.alpha * self.l1_ratio * np.sum(np.abs(self.w_))

        l2_ratio = 1 - self.l1_ratio
        l2_add = self.alpha * l2_ratio * np.sum((self.w_) ** 2)
        return super()._count_loss(y_true, y_pred) + l1_add + l2_add

    def _count_grad(self, X, y, y_pred):
        l2_add = 2 * self.alpha * self.w_ * (1 - self.l1_ratio)
        return super()._count_grad(X, y, y_pred) + l2_add

    def _grad_step(self, dfdw, dfdb):
        self.w_ = self._soft_threshold(self.w_ - self.eta * dfdw)
        self.b_ -= self.eta * dfdb
