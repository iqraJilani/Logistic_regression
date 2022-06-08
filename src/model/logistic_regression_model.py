import numpy as np

from .base_model import Model


class LogisticRegression(Model):

    def __init__(self):
        super().__init__()

    def initialize_weights(self):
        """
        Random initialization of weights matrix

        Args:
            n_features (int): no of features of input data, first dimension of weight matrix

        Returns:
            float- numpy matrix: matrix of weights initialized with random float numbers


        """
        self.weights = np.random.randn(self.n_features, 1)
        self.bias = 0

    def forward_pass(self ):
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

    def back_prop(self):
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

    def update_params(self):
        """
        update the weights using the gradients computed for successive steps of gradient descend

        Returns:
            object: self, model object

        """
        print("Grads Shape: ", self.grads["d_weights"].shape)
        self.weights = self.weights - (self.learning_rate * self.grads["d_weights"])
        self.bias = self.bias - (self.learning_rate * self.grads["d_weights"])
        return self

    def infer(self, data, weights, bias, learning_rate):
        super(LogisticRegression, self.__class__).model_data.fset(self, [weights, bias, learning_rate])
        self.data = data
        self.forward_pass()

