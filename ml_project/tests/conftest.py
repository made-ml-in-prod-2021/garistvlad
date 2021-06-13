from typing import List

import pytest
import pandas as pd
import numpy as np

from src.params.feature_params import FeatureParams


@pytest.fixture()
def sample_data():
    sample_size = 100
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(29, 77, size=sample_size),
        'cp': np.random.randint(4, size=sample_size),
        'sex': np.random.randint(2, size=sample_size),
        'trestbps': np.random.randint(94, 200, size=sample_size),
        'chol': np.random.randint(126, 564, size=sample_size),
        'fbs': np.random.randint(2, size=sample_size),
        'restecg': np.random.randint(2, size=sample_size),
        'thalach': np.random.randint(71, 202, size=sample_size),
        'exang': np.random.randint(2, size=sample_size),
        'oldpeak': 4 * np.random.rand(),
        'slope': np.random.randint(3, size=sample_size),
        'ca': np.random.randint(5, size=sample_size),
        'thal': np.random.randint(4, size=sample_size),
        'target': np.random.randint(2, size=sample_size),
    })
    return df


@pytest.fixture()
def categorical_features() -> List[str]:
    return ["cp", "restecg", "slope", "ca", "thal"]


@pytest.fixture()
def binary_features() -> List[str]:
    return ["sex", "fbs", "exang"]


@pytest.fixture()
def numeric_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture()
def target_col() -> str:
    return "target"


@pytest.fixture()
def feature_params(
        categorical_features: List[str],
        binary_features: List[str],
        numeric_features: List[str],
        target_col: str
) -> FeatureParams:
    params = FeatureParams(
        binary_features=binary_features,
        categorical_features=categorical_features,
        numerical_features=numeric_features,
        target_col=target_col
    )
    return params
