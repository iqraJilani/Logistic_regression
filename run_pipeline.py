from src.pipeline.logistic_regression import RegressionPipeline
import numpy as np

transformations = {
    "standard_scaler": ['car_ID', 'symboling', 'wheelbase', 'carlength', 'carwidth',
       'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',
       'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg',
       'price']
}
rg_pipeline = RegressionPipeline("/home/iqra/RevolveAI/Deep_Learning/Logistic_regression/data", "CarPrice_Assignment",
                                 "price")
rg_pipeline.pre_processing(transformations)
cost_train = rg_pipeline.train_model(0.3, 40)
cost_test = rg_pipeline.evaluate_model()
print(f"This model's training performance is {cost_train}")
