import numpy as np


class BuildModel:
    def __init__(self):
        self.grads = None
        self.bias = None
        self.weights = None
        self.predictions = None
        self.target = None
        self.data = None
        self.n_features = None
        self.n_examples = None
        self.learning_rate = None
        # self.params = {
        #     "weights": self.weights,
        #     "bias": self.bias,
        #     "leaning_rate": self.leaning_rate,
        # }

    def set_data(self, learning_rate, data, target):
        self.learning_rate = learning_rate
        self.target = target
        self.data = data
        self.n_examples, self.n_features = self.data.shape
        self.weights, self.bias = BuildModel.initialize_weights(self.n_features)

    def get_data(self):
        return tuple(self.weights, self.bias, self.learning_rate)

    model_data = property(set_data, get_data)

    @staticmethod
    def initialize_weights(n_features):
        weights = np.random.randn(n_features, 1)
        bias = 0
        return weights, bias

    @staticmethod
    def custom_sigmoid(activations):
        exponents = np.exp(-activations)
        sigmoids = 1 / (1 + exponents)
        return sigmoids

    @staticmethod
    def custom_softmax(probability):
        probability[probability >= 0.7] = 1
        probability[probability < 0.7] = 0
        predictions = probability.astype(int)
        return predictions

    def predict(self, ):
        activations = np.dot(self.data, self.weights)
        probabilities = BuildModel.custom_sigmoid(activations)
        predictions = BuildModel.custom_softmax(probabilities)
        self.predictions = predictions
        print("predictions shape", self.predictions.shape)
        return self

    def compute_cost(self):
        print("predictions shape: ", self.predictions.shape)
        print("target shape: ", self.target.shape)
        epsilon = 1e-5
        cost = np.sum(
            (self.target * np.log(self.predictions + epsilon))
            + ((1 - self.target) * np.log(1 - self.predictions + epsilon))
        )
        cost = -(1 / self.n_examples) * cost
        return cost

    def compute_grad(self):
        d_activations = self.predictions - self.target
        print("d_activations shape", d_activations.shape)
        d_weights = (1 / self.n_examples) * (np.dot(self.data.T, d_activations))
        d_bias = (1 / self.n_examples) * np.sum(d_activations)
        self.grads = {"d_weights": d_weights, "d_bias": d_bias}
        return self

    def update_params(self):
        print("Grads Shape: ", self.grads["d_weights"].shape)
        self.weights = self.weights - (self.learning_rate * self.grads["d_weights"])
        self.bias = self.bias - (self.learning_rate * self.grads["d_weights"])
        return self
