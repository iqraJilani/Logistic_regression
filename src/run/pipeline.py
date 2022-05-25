from src.dataloader import pre_processing
from src.model import build_model


class Pipeline:
    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = None

    def pre_process(self):
        pre_process_obj = pre_processing.PreProcess(self.data)
        pre_process_obj.fix_null()
        pre_process_obj.encode_data()
        self.X_train, self.X_test, y_train, y_test = pre_process_obj.split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, X_train, y_train, learning_rate, num_iter):
        model_obj = build_model.BuildModel(self.X_train)
        weights, bias = model_obj.initialize_weights()
        for i in range(0, num_iter):
            y_predict = model_obj.predict(weights, bias)
            cost = model_obj.compute_cost(y_predict, y_train)
            grads = model_obj.compute_grad(y_predict, y_train)
            weights, bias = model_obj.update_params(weights, bias, learning_rate, grads)
            print(f"Cost in Iteration{i} is: {cost}")
        return weights, bias, learning_rate, cost

    def test_model(self, weights, bias, X_test, y_test, learning_rate):
        model_obj = build_model.BuildModel(X_test)
        y_predict = model_obj.predict(weights, bias)
        cost = model_obj.compute_cost(y_predict, y_test)
        return cost

    def evaltuate_model(self, cost_train, cost_test):
        pass

