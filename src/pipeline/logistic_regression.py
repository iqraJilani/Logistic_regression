from src.dataloader.pre_processing import PreProcess
from src.model.build_model import BuildModel
from src.dataloader.loader import Loader


class RegressionPipeline:

    def __init__(self, path, name, target):
        self.path = path
        self.name = name
        self.target = target
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.learning_rate = None
        self.model_obj = BuildModel()

    def pre_processing(self, transformations):
        loader = Loader(self.path, self.name, self.target)
        loader.pre_process(transformations)
        self.X_train, self.X_test, self.y_train, self.y_test = loader.split_data("default")

    def train_model(self, learning_rate, num_iter):
        self.learning_rate = learning_rate
        self.model_obj.set_data(self.learning_rate, self.X_train, self.y_train)
        for i in range(0, num_iter):
            cost_train = self.model_obj.predict().compute_cost()
            self.model_obj.compute_grad().update_params()
        return cost_train

    def evaluate_model(self):
        self.model_obj.set_data(self.learning_rate, self.X_test, self.y_test)
        cost_test = self.model_obj.predict().compute_cost()
        return cost_test
