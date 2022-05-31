from model import Model
import numpy as np


class LogisticRegression(Model):

    @staticmethod
    def initialize_weights(n_features):
        """
        Random initialization of weights matrix

        Args:
            n_features (int): no of features of input data, first dimension of weight matrix

        Returns:
            float- numpy matrix: matrix of weights initialized with random float numbers


        """
        weights = np.random.randn(n_features, 1)
        bias = 0
        return weights, bias

    def forward_pass(self, ):
        """
        calculate predictions for target variable using input data and weights

        Returns:
            object: self, model object

        """
        activations = self.linear_forward(self.data, self.weights)
        probabilities = self.custom_sigmoid(activations)
        predictions = self.custom_softmax(probabilities)
        self.predictions = predictions
        print("predictions shape", self.predictions.shape)
        return self

    def compute_grad(self):
        """
        compute gradients for gradient descend of optimization of cost on training data

        Returns:
            object: self, model object

        """
        d_activations = self.predictions - self.target
        print("d_activations shape", d_activations.shape)
        d_weights = (1 / self.n_examples) * (np.dot(self.data.T, d_activations))
        d_bias = (1 / self.n_examples) * np.sum(d_activations)
        self.grads = {"d_weights": d_weights, "d_bias": d_bias}
        return self


