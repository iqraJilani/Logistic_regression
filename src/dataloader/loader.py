from pre_processing import PreProcess
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class Loader:
    def __init__(self, path, file_name, target):
        self.data = pd.read_csv(os.path.join(path, file_name))
        self.target = self.data.loc[:, target]

    def pre_process(self, **methods):
        pre_obj = PreProcess(self.data)
        for transformation, cols, method in methods.items():
            pre_obj.fit_transform(transformation, cols)

        self.data = pre_obj.data

    def split_data(self, split_method):
        if split_method == "default":
            data_train, data_test, target_train, target_test = train_test_split(self.data, self.target, test_size=0.3)
            return data_train, data_test, target_train, target_test
