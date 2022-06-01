from .pre_processing import PreProcess
import pandas as pd
import os

import numpy as np

class Loader:
    def __init__(self, path, file_name, target_col):
        self.target_col = target_col
        self.path = path
        self.file_name = file_name
        self.target = None
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.path, self.file_name))
        print("original data shape: ", self.data.shape)
        self.target = self.data.loc[:, self.target_col]
        self.target = np.array([self.target]).reshape(-1, 1)
        self.data = self.data.loc[:, self.data.columns != self.target_col]
        print("X shape: ", self.data.shape)
        print("Y shape: ", self.target.shape)





