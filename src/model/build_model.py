import numpy as np


class BuildModel:
    def __init__(self, input_data, target, leaning_rate):
        self.n_examples, self.n_features = input_data.shape

        self.leaning_rate = leaning_rate
        self.data = input_data
        print(self.data.shape)
        self.target = target
        self.predictions = None
        self.costs = list()
        self.grads = list()
        self.weights = self.initialize_weights()
        self.bias = 0
        self.params = {
            "predictions": self.predictions,
            "weights": self.weights,
            "bias": self.bias,
            "costs": self.costs,
            "grads": self.grads,
        }

    @staticmethod
    def initialize_weights(self):
        weights = np.random.randn(self.n_features, 1)
        return weights

    def custom_sigmoid(self, activations):
        exponents = np.exp(-activations)
        sigmoids = 1 / (1 + exponents)
        return sigmoids

    def custom_softmax(self, probability):
        probability[probability >= 0.7] = 1
        probability[probability < 0.7] = 0
        self.predictions = probability.astype(int)
        return self.predictions

    def predict(self):
        print("inside predict")
        print(self.data.shape)
        print(self.weights.shape)
        activations = np.dot(self.data, self.weights)
        probabilities = self.custom_sigmoid(activations)
        self.custom_softmax(probabilities)
        return self

    def compute_cost(self):
        epsilon = 1e-5
        cost = np.sum(
            (self.target * np.log(self.predictions + epsilon))
            + ((1 - self.target) * np.log(1 - self.predictions + epsilon))
        )
        cost = -(1 / self.n_examples) * cost
        self.costs.append(cost)
        return self

    def compute_grad(self):
        d_activations = self.predictions - self.target
        d_weights = (1 / self.n_examples) * (np.dot(self.data.T, d_activations))
        d_bias = (1 / self.n_examples) * np.sum(d_activations)
        grads = {"d_weights": d_weights, "d_bias": d_bias}
        self.grads.append(grads)
        return self

    def update_params(self, i):
        self.weights = self.weights - (self.leaning_rate * self.grads[i]["d_weights"])
        self.bias = self.bias - (self.leaning_rate * self.grads[i]["d_weights"])
        return self

    def output(self):
        return self.params
