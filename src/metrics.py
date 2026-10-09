import numpy as np


# Метрики регрессии
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    residual_sum = np.sum((y_true - y_pred)**2)
    total_sum = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (residual_sum / total_sum)
