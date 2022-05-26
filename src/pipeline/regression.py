from src.dataloader.pre_processing import PreProcess
from src.model.build_model import BuildModel


class Regression:
    def __init__(self, data, y):
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.data = data
        self.y = y
        # self.X_train, self.X_test, self.y_train, self.y_test = None

    def pre_process(self):
        pre_process_obj = PreProcess(self.data)
        pre_process_obj.fix_null()
        pre_process_obj.encode_data()
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = pre_process_obj.split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, X_train, y_train, learning_rate, num_iter):
        model_obj = BuildModel(X_train, y_train, learning_rate)
        for i in range(0, num_iter):
            model_obj.predict().compute_cost().compute_grad().update_params(i)
        return_data = model_obj.output()
        return return_data["weights"], return_data["bias"], return_data["costs"]

    def test_model(self, weights, bias, X_test, y_test, learning_rate):
        model_obj = BuildModel(X_test)
        y_predict = model_obj.predict(weights, bias)
        cost = model_obj.compute_cost(y_predict, y_test)
        return cost

    def evaltuate_model(self, cost_train, cost_test):
        pass
