from src.pipeline.classification_pipeline import ClassificationPipeline
from src.dataloader.loader import Loader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

scaling = "min_max_scaler"
data_path = "./data"
file_name = "bank_data.csv"
target_var = "approved"
model_type = "neural"
learning_rate = 0.07
task = "inference"

# k = ['k', 'job', 'marital', 'education', 'default', 'housing', 'loan',
#        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
#        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
#        'cons.conf.idx', 'euribor3m', 'nr.employed']
# vals = [
#     [38,'admin','married','high.school','unknown','no','no','telephone','may','mon',165,2,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [49,'entrepreneur','married','university.degree','unknown','yes','no','telephone','may','mon',1042,1,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [38,'technician','single','university.degree','no','no','yes','telephone','may','mon',20,1,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [31,'admin','divorced','high.school','no','no','no','telephone','may','mon',246,1,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [41,'management','married','basic.6y','no','no','no','telephone','may','mon',529,2,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [39,'admin','married','university.degree','no','yes','yes','telephone','may','mon',192,1,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191],
#     [49,'technician','married','basic.9y','no','no','no','telephone','may','mon',1467,1,999,0,'nonexistent',1.1,93.994,-36.4,4.857,5191]
# ]
#
# infer_data = pd.DataFrame(dict(zip(k, vals)))
dd = {
    'k': [38,49],
    'job': ['admin.', 'entrepreneur'],
    'marital': ['married', 'married'],
    'education':['high.school','university.degree'],
    'default': ['unknown', 'unknown'],
    'housing': ['no', 'yes'],
    'loan': ['no', 'no'],
    'contact': ['telephone', 'telephone'],
    'month': ['may', 'may'],
    'day_of_week': ['mon', 'mon'],
    'duration': [165, 1042],
    'campaign': [2, 1],
    'pdays': [999, 999],
    'previous': [0, 0],
    'poutcome': ['nonexistent', 'nonexistent'],
    'emp.var.rate': [1.1, 1.1],
    'cons.price.idx': [93.994, 93.994],
    'cons.conf.idx': [-36.4, -36.4],
    'euribor3m': [4.857,4.857],
    'nr.employed': [5191, 5191]
}
#infer_data = pd.DataFrame(test_data, index=[i for i in range(0, test_data.shape[0])], columns=['c'+str(i) for i in range(0, test_data.shape[1])])
#print(infer_data.dtypes)

infer_data = pd.DataFrame(dd)

rg_pipeline = ClassificationPipeline(model_type, learning_rate)

if task == "training and evaluation":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing(scaling, "fit_transform", data, target)
    rg_pipeline.split_data("default")
    cost_train, costs = rg_pipeline.train_model(300)
    rg_pipeline.save_hyperparameters()
    cost_test = rg_pipeline.evaluate_model()
    print(f"This model's training performance is {cost_train}")
    print(f"This model's testing performance is {cost_test}")
    sns.lineplot(x=costs.keys(), y=costs.values())
    plt.show()

elif task == "evaluate only":
    loader = Loader(data_path, file_name, target_var)
    data, target = loader.load_data()
    rg_pipeline.pre_processing(scaling, "transform", data, target)
    weights, bias, learning_rate = rg_pipeline.load_hyperparameters()
    cost_test = rg_pipeline.evaluate_model(weights, bias, learning_rate, external_weights=True, evaluate_only=True)
    print(f"This model's testing performance is {cost_test}")

elif task == "inference":
    rg_pipeline.pre_processing(scaling, "transform", infer_data)
    weights, bias, learning_rate = rg_pipeline.load_hyperparameters()
    predictions = rg_pipeline.inference(weights, bias, learning_rate)
    print([predictions])
