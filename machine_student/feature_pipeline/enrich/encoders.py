from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, PowerTransformer,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler)

class Encoder:
    CATALOGUE = {
        "one_hot": OneHotEncoder,
        "ordinal": OrdinalEncoder,
        "power": PowerTransformer,
        "standard": StandardScaler,
        "min_max": MinMaxScaler,
        "max_abs": MaxAbsScaler
    }
    def __init__(self, name, encoder_kwargs):
        self.name = name
        self.encoder_kwargs = encoder_kwargs
    
    def get_estimator(self):
        encoder_class = Encoder.CATALOGUE[self.name]
        encoder_estimator = encoder_class(**self.encoder_kwargs)
        return encoder_estimator