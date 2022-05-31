from src.pipeline.classification_pipeline import ClassificationPipeline
import numpy as np

transformations = {
    "standard_scaler": ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
}
data_path = "/home/iqra/RevolveAI/Deep_Learning/Logistic_regression/data"
file_name = "input.csv"
target_var = "Type"
model_type = "logistic"
rg_pipeline = ClassificationPipeline(data_path, file_name, target_var, model_type)
rg_pipeline.pre_processing(transformations)
cost_train = rg_pipeline.train_model(0.3, 40)
cost_test = rg_pipeline.evaluate_model()
print(f"This model's training performance is {cost_train}")
print(f"This model's testing performance is {cost_test}")
