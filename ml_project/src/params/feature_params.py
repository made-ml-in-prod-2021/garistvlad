from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    binary_features: Optional[List[str]]
    categorical_features: Optional[List[str]]
    numerical_features: Optional[List[str]]
    target_col: Optional[str]
