import os
from pathlib import Path

from src.dataloader.pre_processing import PreProcess
from src.model.base_model import Model
from src.model.logistic_regression_model import LogisticRegression
from src.model.neural_network_model import NeuralNetwork
from sklearn.model_selection import train_test_split
import pickle


class ClassificationPipeline:

    def __init__(self, model_type, learning_rate,  n_layers=2, n_nodes=[3, 1]):
        self.target = None
        self.data = None
        self.data_train = None
        self.data_test = None
        self.target_train = None
        self.target_test = None
        self.learning_rate = learning_rate
        self.costs = dict()
        if model_type == "logistic":
            self.model_obj = LogisticRegression()
        elif model_type == "neural":
            self.model_obj = NeuralNetwork(n_layers, n_nodes)

    def pre_processing(self, task, transformations, data, target=None):
        """
        Apply the transformations and other methods e.g., fix skew to the given columns

        Args:
            target ():
            data ():
            methods (dict): transformations and other methods to apply to given list of columns
        """

        pre_obj = PreProcess(data)
        pre_obj.fit_transform(transformations, task)
        self.data = pre_obj.data


        return self

    def split_data(self, split_method):
        """
        split the data into train/test or train/dev/test according to method given

        Args:
            split_method ():

        Returns:
            dataframes & arrays: sectioned data

        """
        if split_method == "default":
            self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(self.data, self.target, test_size=0.3)
            print(self.data_train.shape)
        return self

    def train_model(self, num_iter):
        """
        train the model i.e., find weights that minimize training cost

        Args:
            learning_rate (float): step size for gradient descend of logistic regression
            num_iter (int): no of iterations in the training process

        Returns:
            float: cost of training data, after all iterations.

        """
        self.model_obj.user_data = [self.learning_rate, self.data_train, self.target_train]
        self.model_obj.initialize_weights()
        for i in range(0, num_iter):
            cost_train = self.model_obj.forward_pass().compute_cost()
            self.model_obj.back_prop().update_params()
            print(f"Training cost in iteration {i} is: {cost_train}")
            self.costs[i] = cost_train
        return cost_train, self.costs

    def evaluate_model(self, weights=None, bias=None, learning_rate=None, external_weights = False, evaluate_only=False):
        """
        evaluate the performance of model on testing data using trained weights

        Returns:
            float: cost on test data, computed using trained weights

        """
        if evaluate_only:
            self.model_obj.user_data = [self.learning_rate, self.data, self.target]
        else:
            self.model_obj.user_data = [self.learning_rate, self.data_test, self.target_test]

        if external_weights:
            self.model_obj.model_data = [weights, bias, learning_rate]
        cost_test = self.model_obj.forward_pass().compute_cost()
        return cost_test


    def inference(self, data, weights, bias, learning_rate):
        """
        method to calculate inference on test data

        Returns: numpy array: mx1 array of predictions

        """
        self.model_obj.infer(data, weights, bias, learning_rate)
        predictions = self.model_obj.user_data
        return predictions

    def save_hyperparameters(self):
        """
        saves trained parameters such as weights, bias and learning rate

        """
        params = self.model_obj.model_data
        path = os.path.join(os.getcwd(), 'data/hyperparameters.pkl')
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    def load_hyperparameters(self):

        path = os.path.join(os.getcwd(), 'data/hyperparameters.pkl')
        with open(path, 'rb') as f:
            params = pickle.load(f)
        weights = params["weights"]
        bias = params["bias"]
        learning_rate = params["learning_rate"]
        return weights, bias, learning_rate







