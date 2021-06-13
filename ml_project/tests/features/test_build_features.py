import pandas as pd
import numpy as np

from src.features.build_features import (
    build_transformer,
    make_features,
    fit_transformer,
    extract_target
)


def test_make_features(sample_data: pd.DataFrame, feature_params):
    transformer = build_transformer(feature_params)
    transformer, data = fit_transformer(transformer, sample_data)
    assert not data.duplicated().any()
    assert "categorical_pipeline" in transformer.get_params()
    assert "numerical_pipeline" in transformer.get_params()
    assert "binary_pipeline" in transformer.get_params()
    features = make_features(transformer, data)
    assert features.shape[0] == data.shape[0]


def test_extract_target(sample_data: pd.DataFrame, feature_params):
    extracted_target = extract_target(sample_data, feature_params)
    assert np.all(np.equal(extracted_target, sample_data[feature_params.target_col]))
