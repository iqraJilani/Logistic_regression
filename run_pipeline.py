from src.pipeline.regression import Regression
import numpy as np

data = np.random.randn(150).reshape(15, 10)
print(data.shape)
y = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1]).reshape(15, 1)
pipeline = Regression(data, y)
# X_train, y_train, X_test, y_test = pipeline.pre_process()
trained_weights, trained_bias, cost_train = pipeline.train_model(data, y, 0.3, 40)
# cost_test = pipeline.test_model(trained_weights, trained_bias)
# eval_score = pipeline.evaluate_model(cost_train, cost_test)
print(f"This model's training performance is {cost_train}")
# print(f"This model's testing performance is {cost_test}")
