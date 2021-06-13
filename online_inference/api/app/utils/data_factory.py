import os
from typing import Dict, Union

import numpy as np
import pandas as pd


def generate_example_from_real_dataset(
        real_dataset_filepath: str
) -> Dict[str, Union[int, float]]:
    """Random select of one item from read dataset
        and return features in HeartRequestModel format
    """
    if not os.path.exists(real_dataset_filepath):
        raise ValueError("The dataset filepath does not exist")
        return
    df = pd.read_csv(real_dataset_filepath)
    df.pop("target")
    single_example = list(
        df.sample(n=1, replace=True).to_dict(orient='index').values()
    )[0]
    return single_example


def generate_example_from_synthetic_dataset() -> Dict[str, Union[int, float]]:
    """Geterate data example using randomly generated data
    Set of data validation rules were specified at DataModel layer.
    """
    single_example = {
        # binary features
        'sex': np.random.randint(2),
        'fbs': np.random.randint(2),
        'exang': np.random.randint(2),
        # categorical features
        'cp': np.random.randint(4),
        'restecg': np.random.randint(3),
        'slope': np.random.randint(3),
        'ca': np.random.randint(5),
        'thal': np.random.randint(4),
        # numerical features
        'age': np.random.randint(10, 101),
        'trestbps': np.random.randint(90, 221),
        'chol': np.random.randint(100, 1001),
        'thalach': np.random.randint(70, 201),
        'oldpeak': np.random.rand() * 5,
    }
    return single_example
