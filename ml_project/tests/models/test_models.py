import os
from typing import Tuple

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.params.train_params import TrainParams, LogisticRegressionParams
from src.features.build_features import (
    build_transformer,
    make_features,
    fit_transformer,
    extract_target
)
from src.models.model_fit_predict import (
    save_model,
    load_model,
    train_model
)


@pytest.fixture
def features_and_target(sample_data: pd.DataFrame, feature_params) -> Tuple[pd.DataFrame, pd.Series]:
    transformer = build_transformer(feature_params)
    transformer, data = fit_transformer(transformer, sample_data)
    features = make_features(transformer, data)
    target = extract_target(data, feature_params)
    return features, target


def test_fit_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    classifier_example = LogisticRegression
    classifier_type = "LogisticRegression"
    features, target = features_and_target
    model = train_model(
        features,
        target,
        TrainParams(
            model_type=classifier_type,
            classifier_params=LogisticRegressionParams(
                C=2,
                max_iter=100,
                random_state=101,
            )
        )
    )
    assert isinstance(model, classifier_example)
    assert model.predict(features).shape[0] == target.shape[0]


def test_save_load_model(tmpdir):
    model = RandomForestClassifier()
    expected_output = tmpdir.join("model.pkl")

    output = save_model(model, expected_output)
    assert output == expected_output
    assert os.path.exists(expected_output)

    model = load_model(output)
    assert isinstance(model, RandomForestClassifier)
