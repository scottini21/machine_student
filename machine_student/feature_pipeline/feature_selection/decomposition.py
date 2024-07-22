from typing import Dict, Any

from sklearn.base import TransformerMixin
from sklearn.decomposition import (PCA, LatentDirichletAllocation, NMF,
                                   TruncatedSVD)


class Decomposition:
    CATALOGUE = {
        "pca": PCA,
        "nmf": NMF,
        "lda": LatentDirichletAllocation,
        "truncated_svd": TruncatedSVD
    }

    def __init__(self,
                 name: str,
                 decomposition_kwargs: Dict[str, Any]
                 ) -> None:
        self.name = name
        self.decomposition_kwargs = decomposition_kwargs

    def get_estimator(self) -> TransformerMixin:
        d_class = Decomposition.CATALOGUE[self.name]
        estimator = d_class(**self.decomposition_kwargs)
        return estimator
