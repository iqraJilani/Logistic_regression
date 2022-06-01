import numpy as np
from .base_model import Model


class NeuralNetwork(Model):
    def __init__(self, n_layers, n_nodes):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.activations = dict()
        super().__init__()
        self.grads = dict()

    def initialize_weights(self):
        """
        Random initialization of weights matrix

        Args:
            n_layers ():
            n_nodes ():
            n_examples ():
            n_features (int): no of features of input data, first dimension of weight matrix

        Returns:
            float- numpy matrix: matrix of weights initialized with random float numbers


        """
        self.weights = dict()
        self.bias = dict()
        self.n_nodes = list([self.n_features]) + self.n_nodes
        for i in range(1, self.n_layers + 1):
            j = i - 1
            n_prev = self.n_nodes[j]
            n_current = self.n_nodes[i]
            self.weights[i] = np.random.randn(n_prev, n_current) * 0.05
            self.bias[i] = np.zeros((self.n_examples, n_current))

    @staticmethod
    def custom_tanh(z):
        activations = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return activations

    def forward_pass(self):
        self.activations[0] = self.data
        for i in range(1, self.n_layers):
            j = i - 1
            linear_activations = super().linear_forward(self.activations[j], self.weights[i])
            self.activations[i] = self.custom_tanh(linear_activations)
        j = self.n_layers - 1
        linear_activations = super().linear_forward(self.activations[j], self.weights[self.n_layers])
        self.activations[self.n_layers] = super().custom_sigmoid(linear_activations)
        self.predictions = super().custom_softmax(self.activations[self.n_layers])

        return self

    def back_prop(self):
        d_z_2 = self.activations[2] - self.target
        d_weights_2 = 1 / self.n_examples * np.dot(self.activations[1].T, d_z_2)
        d_bias_2 = 1 / self.n_examples * np.sum(self.bias[2], axis=0, keepdims=True)

        d_g_1 = 1 - np.square(self.activations[1])
        d_z_1 = np.dot(d_z_2, self.weights[2].T) * d_g_1
        d_weights_1 = 1 / self.n_examples * np.dot(self.data.T, d_z_1)
        d_bias_1 = 1 / self.n_examples * np.sum(self.bias[1], axis=0, keepdims=True)

        d_weights = {
            1: d_weights_1,
            2: d_weights_2
        }
        d_bias = {
            1: d_bias_1,
            2: d_bias_2
        }

        self.grads["d_weights"] = d_weights
        self.grads["d_bias"] = d_bias

        return self

    def update_params(self):
        """
        update the weights using the gradients computed for successive steps of gradient descend

        Returns:
            object: self, model object

        """
        for i in range(1, self.n_layers + 1):
            self.weights[i] = self.weights[i] - (self.learning_rate * self.grads["d_weights"][i])
            self.bias[i] = self.bias[i] - (self.learning_rate * self.grads["d_bias"][i])
        return self

    def infer(self, data):
        self.data = data
        self.forward_pass()
        return self.predictions
