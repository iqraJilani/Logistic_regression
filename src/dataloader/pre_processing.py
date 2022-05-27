from sklearn.preprocessing import OneHotEncoder


class PreProcess:
    def __init__(self, data):
        self.data = data

    def fix_skew(self):
        pass

    def fix_null(self):
        pass

    def encode_data(self, method):
        pass

    def split_data(self, method):
        pass
