
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
import os
import pickle


class PreProcess:
    def __init__(self, data):
        self.data = data
        self.task = None
        self.transformers = {
            "null_values": [self.fix_null],
            "skew": [self.fix_skew, PowerTransformer(method='box-cox')],
            "label_encoder": [self.label_encode, LabelEncoder()],
            "one_hot_encoder": [self.one_hot_encode, OneHotEncoder()],
            "min_max_scaler": [self.min_max_scaler, MinMaxScaler()],
            "standard_scaler": [self.standard_scaler, StandardScaler()],
        }

    def fix_skew(self, target, cols):
        """
        fix data skewness

        Args:
            target (string): name of column (target to predict) to drop whilst evaluating skewness
            cols (list): list of columns to apply the transformation to
        """
        skew_feats = self.data.drop(target, axis=1).skew().sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skew_feats})
        skewness = skewness[abs(skewness) > 0.5].dropna()
        skewed_columns = skewness.index

        self.transformers["skew"][1].fit(self.data[skewed_columns])
        self.data[skewed_columns] = self.transformers["skew"][1].transform(self.data[skewed_columns])

    def fix_null(self):
        """
        fix null values in input data

        """
        pass

    def min_max_scaler(self, cols):
        """
        Scale the data in range of 0 to 1

        Args:
            cols (list): list of columns to apply the transformation to
        """

        if self.task == "fit":
            self.transformers["min_max_scaler"][1].fit(self.data[cols].values)
        elif self.task == "transform":
            scaled_cols = self.transformers["min_max_scaler"].transform(self.data[cols].values)
            self.data[cols] = scaled_cols
        elif self.task == "fit_transform":
            self.transformers["min_max_scaler"][1].fit(self.data[cols].values)
            scaled_cols = self.transformers["min_max_scaler"].transform(self.data[cols].values)
            self.data[cols] = scaled_cols


    def standard_scaler(self, cols):
        """
        Standardize the numeric data to have zero mean and standard deviation of 1

        Args:
            cols (): list of columns to apply the transformation to
        """

        if self.task == "fit":
            self.transformers["standard_scaler"][1].fit(self.data[cols].values)
        elif self.task == "transform":
            scaled_cols = self.transformers["min_max_scaler"].transform(self.data[cols].values)
            self.data[cols] = scaled_cols
        elif self.task == "fit_transform":
            self.transformers["min_max_scaler"][1].fit(self.data[cols].values)
            scaled_cols = self.transformers["min_max_scaler"].transform(self.data[cols].values)
            self.data[cols] = scaled_cols

    def label_encode(self, cols):
        """
        Apply label encoding to categorical data

        Args:
            cols (): list of columns to apply the transformation to
        """
        self.data[cols] = self.data[cols].apply(lambda col: self.transformers["label_encoder"].fit_transform(col))

    def one_hot_encode(self, cols):
        """
        Apply one hot encoding to categorical data

        Args:
            cols (): list of columns to apply the transformation to
        """
        # self.data[cols] = self.data[cols].apply(lambda col: self.transformers["one_hot_encoder"].fit_transform(col))
        self.data = pd.get_dummies(self.data, columns=cols)

    def fit_transform(self, transformations, task):
        """
        function to fit the transformers to training data and transform training and test data

        Args:
            transformation (dict): transformations to apply to given list of columns
            cols (list): list of columns to apply transformations to
        """
        self.task = task
        if task == "fit" or "fit_transform":
            for transformation, cols in transformations.items():
                self.transformers[transformation][0](cols)
            with open(os.path.join(os.getcwd(), 'data/transformers.pkl'), 'wb') as f:
                pickle.dump(self.transformers, f)

        elif task == "transform":
            with open(os.path.join(os.getcwd(), 'data/transformers.pkl'), 'rb') as f:
                self.transformers = pickle.load(f)
            for transformation, cols in transformations.items():
                self.transformers[transformation][0](cols)


