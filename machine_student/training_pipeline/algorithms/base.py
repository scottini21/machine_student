from machine_student.algorithms.classification import ClassificationAlgorithm
from machine_student.algorithms.regression import RegressionAlgorithm
from machine_student.algorithms.clustering import ClusteringAlgorithm


class Algorithm:
    TASK_CATALOGUE = {
        "classification": ClassificationAlgorithm,
        "regression": RegressionAlgorithm,
        "clustering": ClusteringAlgorithm
        }

    def __init__(self, task, name, algorithm_kwargs):
        self.task = task
        self.name = name
        self.algorithm_kwargs = algorithm_kwargs

    def get_task_type_algorithms(self):
        return Algorithm.TASK_CATALOGUE[self.task]

    def get_estimator(self):
        task_class = Algorithm.TASK_CATALOGUE[self.task]
        algorithm = task_class(self.name, self.algorithm_kwargs)
        algorithm_estimator = algorithm.get_estimator()
        return algorithm_estimator
