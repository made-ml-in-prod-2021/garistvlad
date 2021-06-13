from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.params.feature_params import FeatureParams


def build_binary_pipeline() -> Pipeline:
    """Build basic pipeline for binary features"""
    return Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ])


def build_categorical_pipeline() -> Pipeline:
    """Build basic pipeline for categorical features"""
    return Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("oh", OneHotEncoder()),
    ])


def build_numerical_pipeline() -> Pipeline:
    """Build basic pipeline for numerical features"""
    return Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaling", StandardScaler()),
    ])


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """Build basic pipeline for set of different features"""
    transformer = ColumnTransformer([
        (
            'binary_pipeline',
            build_binary_pipeline(),
            params.binary_features
        ),
        (
            'categorical_pipeline',
            build_categorical_pipeline(),
            params.categorical_features
        ),
        (
            'numerical_pipeline',
            build_numerical_pipeline(),
            params.numerical_features,
        ),
    ])
    return transformer


def fit_transformer(
        transformer: ColumnTransformer,
        data: pd.DataFrame
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """Fit transformer"""
    without_duplicate_data = data.drop_duplicates()
    without_duplicate_data.reset_index(drop=True, inplace=True)
    transformer.fit(data)
    return transformer, without_duplicate_data


def make_features(
        transformer: ColumnTransformer,
        data: pd.DataFrame
) -> pd.DataFrame:
    """Apply fitted transformer to process data"""
    return transformer.transform(data)


def extract_target(
        data: pd.DataFrame, params: FeatureParams
) -> pd.Series:
    """Get target column from dataset"""
    return data[params.target_col]
