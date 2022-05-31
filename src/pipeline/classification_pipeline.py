from src.dataloader.pre_processing import PreProcess
from src.model.model import Model
from src.model.logistic_regression_model import LogisticRegression
from src.model.neural_network_model import NeuralNetwork
from src.dataloader.loader import Loader


class ClassificationPipeline:

    def __init__(self, path, name, target, model_type):
        self.path = path
        self.name = name
        self.target = target
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.learning_rate = None
        if model_type == "logistic":
            self.model_obj = LogisticRegression()
        else:
            self.model_obj = NeuralNetwork()


    def pre_processing(self, transformations):
        """
        Apply pre-processing to the data and split it into train/test or train/dev/test

        Args:
            transformations (dict): transformations to apply to given list of columns
        """
        loader = Loader(self.path, self.name, self.target)
        loader.pre_process(transformations)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.split_data("default")

    def train_model(self, learning_rate, num_iter):
        """
        train the model i.e., find weights that minimize training cost

        Args:
            learning_rate (float): step size for gradient descend of logistic regression
            num_iter (int): no of iterations in the training process

        Returns:
            float: cost of training data, after all iterations.

        """
        self.learning_rate = learning_rate
        self.model_obj.set_data(self.learning_rate, self.X_train, self.y_train)
        for i in range(0, num_iter):
            cost_train = self.model_obj.forward_pass().compute_cost()
            self.model_obj.compute_grad().update_params()
            print(f"Training cost in iteration {i} is: {cost_train}")
        return cost_train

    def evaluate_model(self):
        """
        evaluate the performance of model on testing data using trained weights

        Returns:
            float: cost on test data, computed using trained weights

        """
        self.model_obj.set_data(self.learning_rate, self.X_test, self.y_test)
        cost_test = self.model_obj.predict().compute_cost()
        return cost_test
