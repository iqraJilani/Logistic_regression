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

    @staticmethod
    def custom_sigmoid(activations):
        """
        calculate sigmoid of activations computed on input data

        Args:
            activations (float-numpy array): numpy array of activations(z) having dimensions of (no_of_examples x 1)

        Returns:
            float-numpy array: element wise sigmoid of the activations(z)  of input data

        """
        exponents = np.exp(-activations)
        sigmoids = 1 / (1 + exponents)
        return sigmoids

    @staticmethod
    def custom_softmax(probability):
        """
        covert probabilities into target categories of 0 and/or 1

        Args:
            probability (float- numpy array): numpy array of probabilities of target being 1 or 0

        Returns:
            numpy array: numpy array of 0 or 1 as predicted class of input examples

        """
        probability[probability >= 0.7] = 1
        probability[probability < 0.7] = 0
        predictions = probability.astype(int)
        return predictions

    def predict(self, ):
        """
        calculate predictions for target variable using input data and weights

        Returns:
            object: self, model object

        """
        activations = np.dot(self.data, self.weights)
        probabilities = BuildModel.custom_sigmoid(activations)
        predictions = BuildModel.custom_softmax(probabilities)
        self.predictions = predictions
        print("predictions shape", self.predictions.shape)
        return self

    def compute_cost(self):
        """
        calculate cost according to logistic regression formula

        Returns:
            object: self, model object

        """
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
