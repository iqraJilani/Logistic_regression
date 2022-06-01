from src.pipeline.classification_pipeline import ClassificationPipeline
from src.dataloader.loader import Loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

transformations = {
    "standard_scaler": ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
}
data_path = "./data"
file_name = "input.csv"
target_var = "Type"
model_type = "neural"
learning_rate = 0.03
task = "training and evaluation"
infer_data = None
external_weights = False
split_size = 0.3

rg_pipeline = ClassificationPipeline(model_type, learning_rate)

if task == "training and evaluation":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing(transformations, data, target).split(split_size)
    cost_train, costs = rg_pipeline.train_model(400)
    rg_pipeline.save_hyperparameters()
    cost_test = rg_pipeline.evaluate_model()
    print(f"This model's training performance is {cost_train}")
    print(f"This model's testing performance is {cost_test}")
    sns.lineplot(x=costs.keys(), y=costs.values())
    plt.show()

elif task =="evaluate only":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing(transformations, data, target)
    weights, bias, learning_rate = rg_pipeline.load_hyperparameters()
    cost_test = rg_pipeline.evaluate_model(weights, bias, learning_rate, external_weights=True, evaluate_only=True)
    print(f"This model's testing performance is {cost_test}")

elif task == "inference":
    rg_pipeline.pre_processing(transformations, data=infer_data)
    predictions = rg_pipeline.inference()





