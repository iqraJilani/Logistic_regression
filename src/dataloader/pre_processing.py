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
        self.num_cols = None
        self.cat_cols = None
        self.transformers = {
            "null_values": self.fix_null,
            "skew": PowerTransformer(method='box-cox'),
            "label_encoder": list(),
            "one_hot_encoder": OneHotEncoder(),
            "min_max_scaler": MinMaxScaler(),
            "standard_scaler": StandardScaler()
        }

    def save_encoders(self):
        with open(os.path.join(os.getcwd(), 'data/transformers.pkl'), 'wb') as f:
            pickle.dump(self.transformers, f)

    def load_encoders(self):
        with open(os.path.join(os.getcwd(), 'data/transformers.pkl'), 'rb') as f:
            self.transformers = pickle.load(f)

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

    def process_num_cols(self, scaling, task):
        num_cols = self.data.select_dtypes(exclude=['object', 'datetime64']).columns

        if len(num_cols) <1:
            return
        if task == "fit" or "fit_transform":
            scaler = self.transformers[scaling]
            scaler.fit(self.data[num_cols])
            self.data[num_cols] = scaler.transform(self.data[num_cols])
            self.save_encoders()
        elif task == "transform":
            self.load_encoders()
            scaler = self.transformers[scaling]
            self.data[num_cols] = scaler.transform(self.data[num_cols])

    def process_cat_cols(self, task):
        cat_cols = list(self.data.select_dtypes(include=['object']).columns)
        print(cat_cols)
        print(self.data[cat_cols[0]].shape)
        n_cat_cols = len(cat_cols)

        if len(cat_cols) < 1:
            return

        if task == "fit" or "fit_transform":
            for i in range(0, n_cat_cols):
                self.transformers['label_encoder'].append(LabelEncoder().fit(self.data[cat_cols[i]]))
                self.data[cat_cols[i]] = self.transformers['label_encoder'][i].transform(self.data[cat_cols[i]])

            # self.transformers['one_hot_encoder'].fit(self.data[cat_cols])
            # self.data[cat_cols] = self.transformers['one_hot_encoder'].transform(self.data[cat_cols])

            # scaler1 = self.transformers["label_encoder"]
            # scaler2 = self.transformers["one_hot_encoder"]
            #
            # scaler1.fit(self.data[cat_cols])
            # self.data[cat_cols] = scaler1.transform(self.data[cat_cols])
            #
            # scaler2.fit(self.data[cat_cols])
            # self.data = scaler2.transform(self.data[cat_cols])

            self.save_encoders()
        elif task == "transform":
            self.load_encoders()

            for i in range(0, n_cat_cols):
                self.data[cat_cols[i]] = self.transformers['label_encoder'][i].transform(self.data[cat_cols[i]])

            # self.data[cat_cols] = self.transformers['one_hot_encoder'].transform(self.data[cat_cols])

            # scaler1 = self.transformers["label_encoder"]
            # scaler2 = self.transformers["one_hot_encoder"]
            #
            # self.data[cat_cols] = scaler1.transform(self.data[cat_cols])
            # self.data = scaler2.transform(self.data[cat_cols])





