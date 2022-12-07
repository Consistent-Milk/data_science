import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))

    for i, val in enumerate(y):
        ary[i, val] = 1

    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_hidden, num_features)
        )

        self.bias_h = np.zeros(num_hidden)

        # Output
        self.weight_out = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_classes, num_hidden)
        )

        self.bias_out = np.zeros(num_classes)

    def forward(self, X):

        z_h = np.dot(X, np.transpose(self.weight_h)) + self.bias_h
        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, np.transpose(self.weight_out)) + self.bias_out
        a_out = sigmoid(z_out)

        return a_h, a_out

    def backward(self, X: np.ndarray, activation_hidden, activation_output, y: np.ndarray):

        # Part One
        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2.0 * (activation_output - y_onehot) / y.shape[0]

        d_a_out__d_z_out = activation_output * (1.0 - activation_output)

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = activation_hidden

        d_loss__dw_out = np.dot(np.transpose(delta_out), d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        # Part Two

        d_z_out__a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = activation_hidden * (1.0 - activation_hidden)

        d_z_h__d_w_h = X

        d_loss__d_w_h = np.dot(
            np.transpose((d_loss__a_h * d_a_h__d_z_h)),
            d_z_h__d_w_h
        )

        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h)
