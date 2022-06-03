from src.pipeline.classification_pipeline import ClassificationPipeline
from src.dataloader.loader import Loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

transformations = {
    "min_max_scaler": ['x0', 'x1', 'x2', 'x3', 'x4']
}
data_path = "./data"
file_name = "iris_data.csv"
target_var = "type"
model_type = "logistic"
learning_rate = 0.03
task = "training and evaluation"
infer_data = np.array([
    [1,	5,	3.3,	1.4,	0.2,	0],
    [1,	7,	3.2,	4.7,	1.4,	1]
])
external_weights = False
split_size = 0.3

rg_pipeline = ClassificationPipeline(model_type, learning_rate)

if task == "training and evaluation":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing("fit_transform", transformations, data, target).split_data("default")
    cost_train, costs = rg_pipeline.train_model(400)
    rg_pipeline.save_hyperparameters()
    cost_test = rg_pipeline.evaluate_model()
    print(f"This model's training performance is {cost_train}")
    print(f"This model's testing performance is {cost_test}")
    sns.lineplot(x=costs.keys(), y=costs.values())
    plt.show()

elif task == "evaluate only":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing("transform", transformations, data, target)
    weights, bias, learning_rate = rg_pipeline.load_hyperparameters()
    cost_test = rg_pipeline.evaluate_model(weights, bias, learning_rate, external_weights=True, evaluate_only=True)
    print(f"This model's testing performance is {cost_test}")

elif task == "inference":
    #rg_pipeline.pre_processing(transformations, infer_data, "transform")
    weights, bias, learning_rate = rg_pipeline.load_hyperparameters()
    predictions = rg_pipeline.inference(infer_data, weights, bias, learning_rate)
    print([predictions])
