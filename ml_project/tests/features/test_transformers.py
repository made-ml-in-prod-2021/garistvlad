import pytest
import pandas as pd
import numpy as np

from src.features.build_features import (
    build_categorical_pipeline,
    build_numerical_pipeline
)


@pytest.fixture()
def fake_categorical_df() -> pd.DataFrame:
    return pd.DataFrame({"cat_feature": [1, 0, np.nan, 2]})


def test_categorical_processing(fake_categorical_df: pd.DataFrame):
    pipeline = build_categorical_pipeline()
    transformed = pd.DataFrame(pipeline.fit_transform(fake_categorical_df).toarray())
    assert transformed.shape[1] == 3
    assert transformed.values.sum() == 4


@pytest.fixture()
def fake_numerical_df() -> pd.DataFrame:
    np.random.seed(123)
    return pd.DataFrame({"num_feature": np.random.randint(29, 77, size=5)})


def test_numerical_features(fake_numerical_df: pd.DataFrame):
    pipeline = build_numerical_pipeline()
    transformed = pd.DataFrame(pipeline.fit_transform(fake_numerical_df))
    assert transformed.shape == fake_numerical_df.shape
