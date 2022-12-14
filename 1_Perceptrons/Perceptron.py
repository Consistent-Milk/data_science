import numpy as np


class Perceptron:

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.RandomState(self.random_state)
        self.w_: np.ndarray = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_: np.ndarray = np.float_(0.0)
        self.errors_: list = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, 0)
