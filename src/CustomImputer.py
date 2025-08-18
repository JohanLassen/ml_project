
# in case ill make some custom changes to the pipeline!


class CustomImputer:
    def __init__(self, my_param=None):
        self.my_param = my_param

    def fit(self, X, y=None):
        # Learn from X
        return self

    def transform(self, X):
        # Transform X
        return X