from __future__ import annotations

from operator import ge

import numpy as np


class LogisticRegressionGD:
    """
    Gradient descent based logistic
    regression classifier.

    Parameters
    ----------

    eta : float
        Learning Rate
    n_iter : int
        Epochs
    random_state : int
        Random number generator seed for
        random weight generation.


    Attributes
    ----------

    w_ : np.ndarray(1-D)
        Weights after training.
    b_ : np.ndarray(Scalar)
        Bias unit after training.
    losses_ : list
        Mean squared error loss function values
        in each epoch

    """

    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1) -> None:
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionGD:
        """
        Fit training data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_examples, n_features]
            Training vectors, where n_examples is the
            number of available data samples and n_features
            is the number of given features of each samples.
        y : np.ndarray, shape = [n_examples]
            Target/Class values.

        Returns
        -------
        self : Instance of LogisticRegressionGD

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_: list = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * \
                np.dot(np.transpose(X), errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = ((-np.dot(y, np.log(output)))
                    - ((np.dot(1 - y, np.log(1 - output))))
                    / X.shape[0])
            self.losses_.append(loss)

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z: float | np.ndarray) -> float | np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(ge(self.activation(self.net_input(X)), 0.5), 1, 0)
