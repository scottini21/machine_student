from typing import TypeVar, Dict, Any

from sklearn.utils.discovery import all_estimators
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


RegressorEstimator = TypeVar("RegressorEstimator",
                             RegressorMixin,
                             XGBRegressor,
                             LGBMRegressor,
                             CatBoostRegressor)


class RegressionAlgorithm:
    """
    This class encapsulates all ML regression algorithms
    from different ML libraries.

    Attributes
    ----------
    name: str
        Name of the regression algorithm.
    regression_kwargs: Dict[str, Any]
        Dictionary containing the hyperparameters names as the keys
        and its values as values.
    """
    SKLEARN_CATALOGUE = []

    def __init__(self,
                 name: str,
                 regression_kwargs: Dict[str, Any]
                 ) -> None:
        """
        RegressionAlgorithm object constructor.

        Parameters
        ----------
        name : str
            Name of the regression algorithm.
        regression_kwargs : Dict[str, Any]
            Dictionary containing the hyperparameters names as the keys
            and its values as values.
        """
        self.name = name
        self.regression_kwargs = regression_kwargs

    def get_sklearn_estimator(self) -> RegressorMixin:
        """
        If the regression name is found among the sklearn models
        it returns the corresponding model with hyperparameters.

        Returns
        -------
        BaseEstimator
            Sklearn's model corresponding with name and hyperparameters.
        """
        regressors = dict(all_estimators(type_filter="regressor"))
        regressor = regressors[self.name](**self.regression_kwargs)
        return regressor

    def get_estimator(self) -> RegressorEstimator:
        """
        This function gets the estimator with the specified hyperparameters.
        This function gets its estimator from sklearn, xgboost, lightgbm and
        catboost libraries.

        Returns
        -------
        RegressorEstimator
            Regression algorithm with its corresponding hyperparameters.

        Raises
        ------
        ValueError
            If name is not found among the options throughs ValueError.
        """
        if self.name in RegressionAlgorithm.SKLEARN_CATALOGUE:
            return self.get_sklearn_estimator()
        elif self.name == "XGBRegressor":
            return XGBRegressor(**self.regression_kwargs)
        elif self.name == "LGBMRegressor":
            return LGBMRegressor(**self.regression_kwargs)
        elif self.name == "CatBoostRegressor":
            return CatBoostRegressor(**self.regression_kwargs)
        else:
            error_message = f"Invalid value of {self.name} for name attribute."
            raise ValueError(error_message)
