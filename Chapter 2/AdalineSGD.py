from __future__ import annotations
from operator import ge

import numpy as np


class AdalineSGD:
    """
    Adaptive Linear Neuron Classifier using 
    Stochastic Gradient Descent.

    Parameters
    ----------

    eta : float
        Learning Rate (between 0.0 and 1.0).

    n_iter : int
        Number of Epochs.

    shuffle : bool (default: True)
        Shuffles training data every epoch 
        if True to prevent cycles.

    random_state : int
        Random number generator seed for 
        random weight initialization.


    Attributes
    ----------

    w_ : np.ndarray(1-D)
        Weights after fitting.

    b_ : np.ndarray(Scalar)
        Bias unit after fitting.

    losses_ : list
        Mean squared error loss function value
        averaged over all training examples in
        each epoch.

    """

    def __init__(self,
                 eta: float = 0.01,
                 n_iter: int = 10,
                 shuffle: bool = True,
                 random_state: int | None = None):

        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def _initialize_weights(self, m: int) -> None:
        """Initialize weigts to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True

    def _shuffle(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return tuple([X[r], y[r]])

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X: np.ndarray) -> np.ndarray:
        """Compute linear activation"""
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class label after unit step"""
        return np.where(ge(self.activation(self.net_input(X)), 0.5), 1, 0)

    def _update_weights(self, xi: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = np.power(error, 2)
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSGD:
        """
        Fit training data.

        Parameters
        ----------

        X : np.ndarray, shape = [n_examples, n_features]
            Training Vectors, where n_examples is the number
            of training examples and n_features is the 
            number of training features.

        y : np.ndarray, shape = [n_examples]
            Target Values(Classes)

        Returns
        -------

        self : AdalineSGD (object)

        """

        self._initialize_weights(X.shape[1])
        self.losses_: list = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X=X, y=y)
            losses: list = []

            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi=xi, target=target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSGD:
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi=xi, target=target)
        else:
            self._update_weights(xi=X, target=y)
        return self
