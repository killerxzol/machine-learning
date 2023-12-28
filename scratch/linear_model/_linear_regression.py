import numpy as np


class BaseLinearRegression:

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._weights = None

    def _fit(self, X, y):
        X = self._intercept_data(X, self.fit_intercept)
        self._weights = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def _predict(self, X):
        X = self._intercept_data(X, self.fit_intercept)
        return self._decision_function(X)

    def _decision_function(self, X):
        return X @ self._weights

    @staticmethod
    def _intercept_data(data, intercept):
        if intercept:
            return np.column_stack((data, np.ones(data.shape[0])))
        return data

    @property
    def weights_(self):
        return self._weights.copy()


class LinearRegression(BaseLinearRegression):

    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)
