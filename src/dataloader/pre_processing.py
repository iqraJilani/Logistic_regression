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
        self.categories = None
        print(self.data.columns)
        self.dtypes = self.data.dtypes
        self.transformers = None

    def set_transformers(self):
        self.transformers = {
            "null_values": self.fix_null,
            "skew": PowerTransformer(method='box-cox'),
            "label_encoder": list(),
            "one_hot_encoder": OneHotEncoder(sparse=False),
            "min_max_scaler": MinMaxScaler(),
            "standard_scaler": StandardScaler()
        }

    def save_encoders(self):
        with open(os.path.join(os.getcwd(), 'data/encoders.pkl'), 'wb') as f:
            pickle.dump(self.transformers, f)

    def load_encoders(self):
        with open(os.path.join(os.getcwd(), 'data/encoders.pkl'), 'rb') as f:
            self.transformers = pickle.load(f)
        print("saved encoders loaded")
        print(self.transformers)

    def fix_null(self):
        """
        fix null values in input data

        """
        pass

    def fix_skew(self):
        """
        fix data skewness

        Args:
            target (string): name of column (target to predict) to drop whilst evaluating skewness
            cols (list): list of columns to apply the transformation to
        """
        skew_feats = self.data.skew().sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skew_feats})
        skewness = skewness[abs(skewness) > 0.4].dropna()
        skewed_columns = skewness.index

        self.transformers["skew"].fit(self.data[skewed_columns])
        self.data[skewed_columns] = self.transformers["skew"].transform(self.data[skewed_columns])

    def fit_transform(self, scaling):
        self.set_transformers()
        num_cols = self.data.select_dtypes(exclude=['object', 'datetime64']).columns
        cat_cols = list(self.data.select_dtypes(include=['object']).columns)
        n_cat_cols = len(cat_cols)

        if len(num_cols) < 1:
            return

        if len(cat_cols)<1:
            return

        self.transformers[scaling] = self.transformers[scaling].fit(self.data[num_cols])
        self.data[num_cols] = self.transformers[scaling].transform(self.data[num_cols])

        for i in range(0, n_cat_cols):
            self.transformers['label_encoder'].append(LabelEncoder().fit(self.data[cat_cols[i]]))
            self.data[cat_cols[i]] = self.transformers['label_encoder'][i].transform(self.data[cat_cols[i]])
        self.transformers['one_hot_encoder'] = self.transformers['one_hot_encoder'].fit(self.data[cat_cols])
        encoded_data = pd.DataFrame(self.transformers['one_hot_encoder'].transform(self.data[cat_cols]))
        encoded_data.columns = self.transformers['one_hot_encoder'].get_feature_names()

        num_data = self.data.drop(cat_cols, axis=1)
        self.data = pd.concat([num_data, encoded_data], axis=1)

        self.save_encoders()

    def transform(self, scaling):
        self.load_encoders()

        print(self.transformers['min_max_scaler'].__dict__)
        print("encoder categories in inference", self.transformers['one_hot_encoder'].categories_)

        num_cols = self.data.select_dtypes(exclude=['object', 'datetime64']).columns
        cat_cols = list(self.data.select_dtypes(include=['object']).columns)
        n_cat_cols = len(cat_cols)

        if len(num_cols) < 1:
            return

        if len(cat_cols) < 1:
            return

        self.data[num_cols] = self.transformers[scaling].transform(self.data[num_cols])

        for i in range(0, n_cat_cols):
            self.data[cat_cols[i]] = self.transformers['label_encoder'][i].transform(self.data[cat_cols[i]])

        encoded_data = pd.DataFrame(self.transformers['one_hot_encoder'].transform(self.data[cat_cols]))
        encoded_data.columns = self.transformers['one_hot_encoder'].get_feature_names()

        num_data = self.data.drop(cat_cols, axis=1)
        self.data = pd.concat([num_data, encoded_data], axis=1)


