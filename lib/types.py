from abc import ABC, abstractmethod
from typing import Any

from skmultilearn.base.problem_transformation import ProblemTransformationBase

class MultiLabelClassifier(ABC, ProblemTransformationBase):
    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

class OptimizationPipeline(ABC):
    @abstractmethod
    def run(self, X: Any, y: Any) -> Any:
        pass
