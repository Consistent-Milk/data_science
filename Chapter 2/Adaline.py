from operator import ge

import numpy as np


class Adaline:
    """ADAptive LInear NEuron classifier(ADALINE)"""

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.RandomState(self.random_state)
        self.w_: np.ndarray = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_: np.ndarray = np.float_(0.0)
        self.losses_: list = []

        for i in range(self.n_iter):
            net_input: np.ndarray = self.net_input(X)
            output: np.ndarray = self.activation(net_input)
            errors: np.ndarray = (y - output)
            self.w_ += self.eta * 2.0 * \
                np.dot(np.transpose(X), errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (np.power(errors, 2)).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(ge(self.activation(self.net_input(X)), 0.5), 1, 0)
