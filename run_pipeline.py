from src.pipeline.logistic_regression import RegressionPipeline
import numpy as np

transformations = {
    "standard_scaler": ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
}
rg_pipeline = RegressionPipeline("/home/iqra/RevolveAI/Deep_Learning/Logistic_regression/data", "input.csv",
                                 "Type")
rg_pipeline.pre_processing(transformations)
cost_train = rg_pipeline.train_model(0.3, 40)
cost_test = rg_pipeline.evaluate_model()
print(f"This model's training performance is {cost_train}")
print(f"This model's testing performance is {cost_test}")
