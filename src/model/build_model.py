import numpy as np


class BuildModel:
    def __init__(self, X):
        self.data = X
        self.params = dict()
        self.n_examples, self.n_features = self.data.shape
        # self.weights = np.zeros((self.n_features, 1))
        # self.bias = 0
        # self.params["weights"] = self.weights
        # self.params["bias"] = self.bias

    def initialize_weights(self):
        weights = np.zeros((self.n_features))
        bias = 0
        return weights,bias

    def custom_sigmoid(self, activations):
        exponents = np.exp(-activations)
        exponents += 1
        sigmoids = 1 / exponents
        return sigmoids

    def custom_softmax(self, probability):
        probability[probability >= 0.7] = 1
        probability[probability < 0.7] = 0
        return probability

    def predict(self, weights, bias):
        activations = np.dot(weights.T, self.data) + bias
        # activations = z
        self.params["activations"] = activations
        probabilities = self.custom_sigmoid(activations)
        y_predict = self.custom_softmax(probabilities)
        return y_predict

    def compute_cost(self, y_predict, y_actual):
        cost = np.sum((y_actual * np.log(y_predict)) + ((1 - y_actual) * np.log(1 - y_predict)))
        cost = - (1 /self.n_examples) * cost
        return cost

    def compute_grad(self, y_predict, y_actual):
        d_activations = y_predict - y_actual
        d_weights = (1 / self.n_examples) * (np.dot(self.data, d_activations.T))
        d_bias = (1 / self.n_examples) * np.sum(d_activations)
        grads = {"d_weights": d_weights,
                 "d_bias": d_bias}
        return grads

    def update_params(self, weights, bias, alpha, grads):
        weights = weights - (alpha * grads["d_weights"])
        bias = bias - (alpha * grads["d_bias"])

        return weights, bias






