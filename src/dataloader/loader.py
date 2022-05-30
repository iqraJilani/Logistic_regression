from .pre_processing import PreProcess
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

class Loader:
    def __init__(self, path, file_name, target):
        self.target = None
        self.data = pd.read_csv(os.path.join(path, file_name))
        print("original data shape: ", self.data.shape)
        self.target = self.data.loc[:, target]
        self.target = np.array([self.target]).reshape(-1, 1)
        self.data = self.data.loc[:, self.data.columns != target]
        print("X shape: ", self.data.shape)
        print("Y shape: ", self.target.shape)

    def pre_process(self, methods):
        pre_obj = PreProcess(self.data)
        for transformation, cols in methods.items():
            pre_obj.fit_transform(transformation, cols)
        self.data = pre_obj.data

    def split_data(self, split_method):
        if split_method == "default":
            data_train, data_test, target_train, target_test = train_test_split(self.data, self.target, test_size=0.3)
            return data_train, data_test, target_train, target_test
