from dataclasses import asdict, dataclass, field
from typing import Union


@dataclass()
class RandomForestClassifierParams:
    n_estimators: int
    max_depth: int
    max_features: float
    max_samples: float
    bootstrap: bool = field(default=True)
    random_state: int = field(default=42)

    def to_dict(self):
        """Get key: value pairs for specified params"""
        return asdict(self)


@dataclass()
class LogisticRegressionParams:
    C: float
    max_iter: int
    random_state: int = field(default=42)

    def to_dict(self):
        """Get key: value pairs for specified params"""
        return asdict(self)


ClassifierParams = Union[
    RandomForestClassifierParams,
    LogisticRegressionParams
]


@dataclass()
class TrainParams:
    model_type: str
    classifier_params: ClassifierParams
