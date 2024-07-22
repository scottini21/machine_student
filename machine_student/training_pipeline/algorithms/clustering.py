from typing import Dict, Any

from sklearn.utils.discovery import all_estimators
from sklearn.base import ClusterMixin


class ClusteringAlgorithm:
    """
    This class encapsulates all ML regression algorithms
    from different ML libraries.

    Attributes
    ----------
    name: str
        Name of the regression algorithm.
    clustering_kwargs: Dict[str, Any]
        Dictionary containing the hyperparameters names as the keys
        and its values as values.
    """
    SKLEARN_CATALOGUE = []

    def __init__(self,
                 name: str,
                 clustering_kwargs: Dict[str, Any]
                 ) -> None:
        """
        ClusteringAlgorithm object constructor.

        Parameters
        ----------
        name : str
            Name of the clustering algorithm.
        clustering_kwargs : Dict[str, Any]
            Dictionary containing the hyperparameters names as the keys
            and its values as values.
        """
        self.name = name
        self.clustering_kwargs = clustering_kwargs


    def get_estimator(self) -> ClusterMixin:
        """
        This function gets the estimator with the specified hyperparameters
        from sklearn.

        Returns
        -------
        ClusterMixin
            Clustering algorithm with its corresponding hyperparameters.

        """
        clusterings = dict(all_estimators(type_filter="cluster"))
        clustering = clusterings[self.name](**self.regression_kwargs)
        return clustering
