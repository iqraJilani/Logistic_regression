
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer



class PreProcess:
    def __init__(self, data):
        self.data = data
        self.transformers = {
            "skew": PowerTransformer(method='box-cox'),
            "label_encoder": LabelEncoder(),
            "one_hot_encoder": OneHotEncoder(),
            "min_max_scaler": MinMaxScaler(),
            "standard_scaler": self.standard_scaler,
            "power_transformer": PowerTransformer()
        }

    def fix_skew(self, target, cols):
        skew_feats = self.data.drop(target, axis=1).skew().sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skew_feats})
        skewness = skewness[abs(skewness) > 0.5].dropna()
        skewed_columns = skewness.index
        self.transformers["skew"].fit(self.data[skewed_columns])
        self.data[skewed_columns] = self.transformers["skew"].transform(self.data[skewed_columns])

    def fix_null(self):
        pass

    def min_max_scaler(self, cols):
        self.transformers["min_max_scaler"].fit(self.data[cols].values)
        scaled_cols = self.transformers["min_max_scaler"].transform(self.data[cols].values)
        self.data[cols] = scaled_cols

    def standard_scaler(self, cols):
        scaler = StandardScaler()
        scaler.fit(self.data[cols].values)
        scaled_cols = scaler.transform(self.data[cols].values)
        # self.transformers["standard_scaler"].fit(self.data[cols].values)
        # scaled_cols = self.transformers["standard_scaler"].transform(self.data[cols].values)
        self.data[cols] = scaled_cols

    def label_encode(self, cols):
        self.data[cols] = self.data[cols].apply(lambda col: self.transformers["label_encoder"].fit_transform(col))

    def one_hot_encode(self, cols):
        # self.data[cols] = self.data[cols].apply(lambda col: self.transformers["one_hot_encoder"].fit_transform(col))
        self.data = pd.get_dummies(self.data, columns=cols)


    def fit_transform(self, transformation, cols):
        self.transformers[transformation](cols)
