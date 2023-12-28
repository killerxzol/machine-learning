import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error, log_loss
from scipy.special import expit


class BaseSGD(metaclass=ABCMeta):

    def __init__(
            self,
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=100,
            tol=1e-3,
            epsilon=0.1,
            random_state=None,
            learning_rate=0.1,
            n_iter_no_change=5,
            optimization=None
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.n_iter_no_change = n_iter_no_change
        self.optimization = optimization
        self.best_loss = None
        self.iter_no_change = None
        self._weights = None
        self._losses = None
        self._cum_gradient = None

    def _partial_fit(self, X, y):
        X = self.intercept_data(X, self.fit_intercept)

        n_samples, n_features = X.shape

        if self._weights is None:
            self._weights = self.initialize_weights(n_features, self.random_state)
            self._losses = []
            if self.optimization == 'adagrad':
                self._cum_gradient = np.zeros(n_features)

        self._fit_regressor(X, y, partial_fit=True)

        return self

    @abstractmethod
    def partial_fit(self, X, y):
        pass

    def _fit(self, X, y):
        X = self.intercept_data(X, self.fit_intercept)

        n_samples, n_features = X.shape

        self._weights = self.initialize_weights(n_features, self.random_state)
        self._losses = []

        if self.optimization == "adagrad":
            self._cum_gradient = np.zeros(n_features)

        rng = np.random.default_rng(seed=self.random_state)

        indices = np.arange(n_samples)

        self.iter_no_change = 0
        self.best_loss = float('Inf')
        for _ in range(self.max_iter):
            idx = rng.choice(indices, size=1)
            self._fit_regressor(X[idx, :], y[idx])

        return self

    @abstractmethod
    def fit(self, X, y):
        pass

    def _fit_regressor(self, X, y, partial_fit=False):
        y_pred = self._decision_function(X)

        loss = self._calculate_loss(y, y_pred)
        self._losses.append(loss)

        if not partial_fit:

            if loss < self.best_loss - self.tol:
                self.iter_no_change += 1
                if self.n_iter_no_change == self.iter_no_change:
                    return
            else:
                self.iter_no_change = 0

            self.best_loss = min(self.best_loss, loss)

        gradient = self._calculate_gradient(X, y, y_pred)
        step = self._calculate_step(gradient)

        self._weights -= step

    @abstractmethod
    def _decision_function(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @property
    def weights_(self):
        return self._weights.copy()

    @property
    def losses_(self):
        return self._losses.copy()

    @abstractmethod
    def _calculate_gradient(self, X, y_true, y_pred):
        pass

    def _calculate_step(self, gradient):
        if not self.optimization:
            return self.learning_rate * gradient
        elif self.optimization == "adagrad":
            self._cum_gradient += gradient ** 2
            return self.learning_rate / np.sqrt(self._cum_gradient + self.epsilon) * gradient
        else:
            raise ValueError("Optimization should be `None` or `adagrad`")

    @staticmethod
    @abstractmethod
    def _calculate_loss(y_true, y_pred):
        pass

    @staticmethod
    def intercept_data(data, intercept):
        if intercept:
            return np.column_stack((data, np.ones(data.shape[0])))
        return data

    @staticmethod
    def initialize_weights(size, random_state):
        return np.random.default_rng(seed=random_state).random(size=size)

    @staticmethod
    def soft_sign(x, eps=1e-6):
        return np.where(np.abs(x) > eps, np.sign(x), x / eps)


class BaseSGDRegressor(BaseSGD):

    @abstractmethod
    def __init__(
            self,
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=100,
            tol=1e-3,
            epsilon=0.1,
            random_state=None,
            learning_rate=0.1,
            n_iter_no_change=5,
            optimization=None
    ):
        super().__init__(
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter_no_change=n_iter_no_change,
            optimization=optimization
        )

    def _predict(self, X):
        X = self.intercept_data(X, self.fit_intercept)
        return self._decision_function(X)

    def _decision_function(self, X):
        return X @ self._weights

    def _calculate_gradient(self, X, y_true, y_pred):
        n_samples, n_features = X.shape
        if not self.penalty:
            return 2 * X.T @ (y_pred - y_true) / n_samples
        elif self.penalty == "l1":
            sign = self.soft_sign(self._weights)
            if self.fit_intercept:
                sign[-1] = 0
            return X.T @ (y_pred - y_true) / n_samples + self.alpha * sign
        elif self.penalty == "l2":
            A = self.alpha * np.eye(n_features)
            if self.fit_intercept:
                A[-1, -1] = 0
            return 2 * (X.T @ X / n_samples + A) @ self._weights - 2 * (X.T @ y_true / n_samples)
        else:
            raise ValueError("Penalty should be `None`, `l1` or `l2`")

    @staticmethod
    def _calculate_loss(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


class SGDRegressor(BaseSGDRegressor):

    def __init__(
            self,
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=100,
            tol=1e-3,
            epsilon=0.1,
            random_state=None,
            learning_rate=0.1,
            n_iter_no_change=5,
            optimization=None
    ):
        super().__init__(
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter_no_change=n_iter_no_change,
            optimization=optimization
        )

    def partial_fit(self, X, y):
        return self._partial_fit(X, y)

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)


class BaseSGDClassifier(BaseSGD):

    @abstractmethod
    def __init__(
            self,
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=100,
            tol=1e-3,
            epsilon=0.1,
            random_state=None,
            learning_rate=0.1,
            n_iter_no_change=5,
            optimization=None
    ):
        super().__init__(
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter_no_change=n_iter_no_change,
            optimization=optimization
        )

    def _predict_proba(self, X):
        X = self.intercept_data(X, self.fit_intercept)
        return self._decision_function(X)

    def _predict(self, X, threshold):
        return self._predict_proba(X) >= threshold

    def _decision_function(self, X):
        return expit(X @ self._weights)

    def _calculate_gradient(self, X, y_true, y_pred):
        n_samples, n_features = X.shape
        gradient = X.T @ (y_pred - y_true) / n_samples
        if not self.penalty:
            return gradient
        elif self.penalty == "l1":
            sign = self.soft_sign(self._weights)
            if self.fit_intercept:
                sign[-1] = 0
            return gradient + self.alpha * sign
        elif self.penalty == "l2":
            weights = self._weights.copy()
            if self.fit_intercept:
                weights[-1] = 0
            return gradient + 2 * self.alpha * weights
        else:
            raise ValueError("Penalty should be `None`, `l1` or `l2`")

    @staticmethod
    def _calculate_loss(y_true, y_pred):
        return log_loss(y_true, y_pred, labels=[0, 1])


class SGDClassifier(BaseSGDClassifier):

    def __init__(
            self,
            penalty="l2",
            alpha=0.0001,
            fit_intercept=True,
            max_iter=100,
            tol=1e-3,
            epsilon=0.1,
            random_state=None,
            learning_rate=0.1,
            n_iter_no_change=5,
            optimization=None
    ):
        super().__init__(
            penalty=penalty,
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            epsilon=epsilon,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter_no_change=n_iter_no_change,
            optimization=optimization
        )

    def partial_fit(self, X, y):
        return self._partial_fit(X, y)

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X, threshold=0.5):
        return self._predict(X, threshold)
