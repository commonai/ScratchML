import math
import numpy as np
from tqdm import tqdm
from .base import BaseModel


class Perceptron(BaseModel):

    def __init__(self, n_iteration=1000, learning_rate=0.1, random_seed=1):
        self.weights = None
        self.bias = None
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate
        self.seed = random_seed
        self.loss = []

    def _init_weights(self, n_features):
        limit = 1 / math.sqrt(n_features)
        random_gen = np.random.RandomState(self.seed)
        # uniform initializations, you may try different ones
        self.weights = random_gen.uniform(-limit, limit, (n_features,))
        self.bias = random_gen.uniform(-limit, limit, (1,))

    def fit(self, X, Y):
        """Fit model on x and y training data

        Args:
            X (np.array): input, shape (n_samples, n_features)
            Y (np.array): output, shape (n_samples)
        """
        n_features = X.shape[1]
        self._init_weights(n_features)

        for _ in tqdm(range(self.n_iteration)):
            # feedforward and predict
            output = self.predict(X)

            # compare
            error = (Y - output)

            # update weight and bias
            gradient_weights = np.dot(error, X)
            self.weights += self.learning_rate * gradient_weights
            gradient_bias = np.sum(error)
            self.bias += self.learning_rate * gradient_bias

            self.loss.append(self.mean_squared_error(Y, output))

    def linear(self, Z):
        """Activation function. return Z.
            Very redudant, but here to be
            consistent with the rest of the framework

        Args:
            Z (np.array): shape (n_features)

        Returns:
            (np.array): shape (n_features)
        """
        return Z

    def mean_squared_error(self, actual, predicted):
        """mean squared error

        Args:
            actual (np.array): shape, (n_features)
            predicted (np.array): shape, (n_features)

        Returns:
            float: mse error value
        """
        return np.mean(np.square((actual - predicted)))

    def _forward(self, X):
        """returns feedforward prediction
        
        Args:
            X ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        output = np.dot(X, self.weights) + self.bias
        output = self.linear(output)
        return output

    def predict(self, X):
        """returns binary classification prediction. using the
            step function.

            1 if X > 0.0
            0 if X < 0.0
        
        Args:
            X (np.array): shape (n_samples, n_features)
        
        Returns:
            np.array: shape (n_samples, 1)
        """
        return np.where(self._forward(X) > 0.0, 1, 0)
