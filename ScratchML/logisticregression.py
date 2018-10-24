import numpy as np
import math
from .base import BaseModel


class LogReg(BaseModel):
    """Binary classification using logistic regression."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.params = None

    def _init_params(self, n_features):
        """Randomly initialize parameters. Parameters will be set to random
            values between [-1/sqrt(N), 1/sqrt(N)]. Parameters shape
            will equal to n_features of input.

            Args:
                n_features (int): number of input features
        """
        # limit values between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features,))

    def sigmoid(self, x):
        """sigmoid activation. squishes all values between 0 and 1

        Args:
            x (np.array)

        Returns:
            (np.array)
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, n_iterations=1000):
        n_features = np.shape(X)[1]
        self._init_params(n_features)
        for _ in range(n_iterations):
            matmul = X.dot(self.params)
            y_pred = self.sigmoid(matmul)
            gradient = -(y - y_pred).dot(X)
            self.params -= self.learning_rate * gradient

    def _predict(self, X):
        matmul = X.dot(self.params)
        y_pred = self.sigmoid(matmul)
        return y_pred

    def predict(self, X):
        return np.round(self._predict(X))
