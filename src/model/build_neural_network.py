import numpy as np
from build_model import BuildModel


class BuildNeuralNetwork(BuildModel):
    def __init__(self, n_layers, n_nodes):
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.activations = dict()
        super().__init__()

    def set_data(self, learning_rate, data, target):
        super(BuildNeuralNetwork, self.__class__).class_data.fset(self, learning_rate, data, target)
        self.n_nodes = list([self.n_features]) + self.n_nodes
        self.weights, self.bias = BuildNeuralNetwork.initialize_weights(self.n_examples, self.n_nodes, self.n_layers)

    def get_data(self):
        return tuple(self.weights, self.bias, self.learning_rate)

    model_data = property(set_data, get_data)

    @staticmethod
    def initialize_weights(n_examples, n_nodes, n_layers):
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
        weights = dict()
        bias = dict()
        for i in range(1, n_layers + 1):
            j = i - 1
            n_prev = n_nodes[j]
            n_current = n_nodes[i]
            weights[i] = np.random.randn((n_prev, n_current)) * 0.01
            bias[i] = np.zeros((n_examples, n_current))

        return weights, bias

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
        self.activations[self.n_layers] = super().custom_sigmoid(self.activations[j], self.weights[self.n_layers])

        self.predictions = super().custom_softmax(self.activations[self.n_layers])

    def back_prop(self):
        pass






