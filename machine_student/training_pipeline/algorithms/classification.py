from typing import TypeVar, Dict, Any

from sklearn.utils.discovery import all_estimators
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


ClassificationEstimator = TypeVar("ClassificationEstimator",
                                  ClassifierMixin,
                                  XGBClassifier,
                                  LGBMClassifier,
                                  CatBoostClassifier)


class ClassificationAlgorithm:
    """
    This class encapsulates all ML classification algorithms
    from different ML libraries.

    Attributes
    ----------
    name: str
        Name of the classification algorithm.
    classification_kwargs: Dict[str, Any]
        Dictionary containing the hyperparameters names as the keys
        and its values as values.
    """
    SKLEARN_CATALOGUE = []

    def __init__(self,
                 name: str,
                 classification_kwargs: Dict[str, Any]
                 ) -> None:
        """
        ClassificationAlgorithm object constructor.

        Parameters
        ----------
        name : str
            Name of the classification algorithm.
        classification_kwargs : Dict[str, Any]
            Dictionary containing the hyperparameters names as the keys
            and its values as values.
        """
        self.name = name
        self.classification_kwargs = classification_kwargs

    def get_sklearn_estimator(self) -> ClassifierMixin:
        """
        If the classification name is found among the sklearn models
        it returns the corresponding model with hyperparameters.

        Returns
        -------
        BaseEstimator
            Sklearn's model corresponding with name and hyperparameters.
        """
        classifiers = dict(all_estimators(type_filter="classifier"))
        classifier = classifiers[self.name](**self.classification_kwargs)
        return classifier

    def get_estimator(self) -> ClassificationEstimator:
        """
        This function gets the estimator with the specified hyperparameters.
        This function gets its estimator from sklearn, xgboost, lightgbm and
        catboost libraries.

        Returns
        -------
        ClassificationEstimator
            Classification algorithm with its corresponding hyperparameters.

        Raises
        ------
        ValueError
            If name is not found among the options throughs ValueError.
        """
        if self.name in ClassificationAlgorithm.SKLEARN_CATALOGUE:
            return self.get_sklearn_estimator()
        elif self.name == "XGBClassifier":
            return XGBClassifier(**self.classification_kwargs)
        elif self.name == "LGBMClassifier":
            return LGBMClassifier(**self.classification_kwargs)
        elif self.name == "CatBoostClassifier":
            return CatBoostClassifier(**self.classification_kwargs)
        else:
            error_message = f"Invalid value of {self.name} for name attribute."
            raise ValueError(error_message)
