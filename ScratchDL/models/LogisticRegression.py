"""
Logistic Regression for binary classification
"""
import math
import numpy as np
from tqdm import tqdm
from .base import BaseModel


class LogisticRegression(BaseModel):
    """Binary classification using logistic regression.
        Can only take class labels 0 and 1
    """

    def __init__(self, n_iterations=100, learning_rate=0.0001, seed=1):
        self.weights = None
        self.bias = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.seed = seed
        self.loss = []

    def _init_weights(self, n_features):
        """Randomly initialize weights and bias. Parameters will be set to random
            values between [-1/sqrt(N), 1/sqrt(N)].

            Args:
                n_features (int): number of input features
        """
        limit = 1 / math.sqrt(n_features)
        random_gen = np.random.RandomState(self.seed)
        # uniform initializations is arbitrary
        # you may try different ones like normal
        self.weights = random_gen.uniform(-limit, limit, (n_features,))
        self.bias = random_gen.uniform(-limit, limit, (1,))

    def fit(self, X, Y):
        """Fit model on X and Y training data

        Args:
            X (np.array): input, shape (n_samples, n_features)
            Y (np.array): output, shape (n_samples)
        """
        n_features = X.shape[1]
        self._init_weights(n_features)

        for _ in tqdm(range(self.n_iterations)):
            # forward pass
            output = self._forward(X)

            # calculate error
            errors = (Y - output)  # shape, (n_samples, 1)

            # backward pass
            # update weights and bias
            gradient_weights = np.dot(errors, X)
            self.weights += self.learning_rate * gradient_weights
            gradient_bias = np.sum(errors)
            self.bias += self.learning_rate * gradient_bias

            # calculate the loss
            loss = self.log_likelihood(Y, output)
            self.loss.append(loss)

    def log_likelihood(self, Y, output):
        """Log likelihood loss function, to measure how well
            the neural network is doing. This is used for
            logistic regression due to it being convexed in shape

        Args:
            Y (np.array): ground truth, shape (n_samples)
            output (np.array): Y pred, shape (n_samples)

        Returns:
            (float): loss value
        """
        cost_1 = np.dot(Y, np.log(output))
        cost_2 = np.dot((1 - Y), np.log(1 - output))
        return -cost_1 - cost_2

    def sigmoid(self, Z):
        """sigmoid activation. squishes all values between 0 and 1

        Args:
            x (np.array)

        Returns:
            (np.array)
        """
        return 1 / (1 + np.exp(-Z))

    def _forward(self, X):
        """Predicts the sigmoid value

        Args:
            X (np.array): input, shape (n_samples, n_features)

        Returns:
            np.array: y pred, shape (n_samples)
        """
        # (n_samples, n_features) x (n_features, 1)
        # feedforward, shape (n_samples, 1)
        output = np.dot(X, self.weights) + self.bias
        # pass feedforward results to activation function
        output = self.sigmoid(output)
        return output

    def predict(self, X):
        """Rounds the sigmoid values to 0 or 1
            
            0 if X < 0.5
            1 if X >= 0.5

        Args:
            X (np.array)

        Returns:
            np.array: np.array of class labels, 0 or 1
        """
        return np.round(self._forward(X))
