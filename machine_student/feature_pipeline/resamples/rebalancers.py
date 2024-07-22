from typing import Dict, Any

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.combine import SMOTETomek


class Rebalancer:
    REBALANCE_METHODS = {
        "smote": SMOTE,
        "smote_tomek": SMOTETomek,
        "tomek_links": TomekLinks,
        "random_over": RandomOverSampler,
        "random_under": RandomUnderSampler,
        "adasyn": ADASYN
    }

    def __init__(self, name: str, rebalancer_kwargs: Dict[str, Any]) -> None:
        self.name = name
        self.rebalancer_kwargs = rebalancer_kwargs

    def get_estimator(self):
        rebalance_class = Rebalancer.REBALANCE_METHODS[self.name]
        rebalance_estimator = rebalance_class(**self.rebalancer_kwargs)
        return rebalance_estimator
