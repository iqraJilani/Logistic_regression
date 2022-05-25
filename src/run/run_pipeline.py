from pipeline import Pipeline

data = None
y = None
pipeline = Pipeline(data, y)
X_train, y_train, X_test, y_test = pipeline.pre_process()
trained_weights, trained_bias, trained_learning_rate, cost_train= pipeline.train_model()
cost_test = pipeline.test_model(trained_weights, trained_bias)

eval_score = pipeline.evaluate_model(cost_train, cost_test)

print(f"This model's training performance is {cost_train}")
print(f"This model's testing performance is {cost_test}")












































Z