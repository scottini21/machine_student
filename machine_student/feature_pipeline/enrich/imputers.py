from typing import Dict, Any

from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer, IterativeImputer

class Imputer:
    CATALOGUE = {
        "simple": SimpleImputer,
        "iterative": IterativeImputer
    }
    def __init__(self,
                 name: str,
                 imputer_kwargs: Dict[str, Any]
                 ) -> None:
        self.name = name
        self.imputer_kwargs = imputer_kwargs
    
    def get_estimator(self) -> TransformerMixin:
        imputer_class = Imputer.CATALOGUE[self.name]
        imputer_estimator = imputer_class(**self.imputer_kwargs)
        return imputer_estimator
        