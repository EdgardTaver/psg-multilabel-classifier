from abc import ABC, abstractmethod
from typing import Any

from skmultilearn.base.problem_transformation import ProblemTransformationBase

class MultiLabelClassifier(ABC, ProblemTransformationBase):
    @abstractmethod
    def fit(self, X: Any, y: Any):
        pass
    
    @abstractmethod
    def predict(self, X: Any):
        pass
