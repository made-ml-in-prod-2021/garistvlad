import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.params.train_params import TrainParams

SklearnClassificationModel = Union[
    RandomForestClassifier,
    LogisticRegression,
]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainParams
) -> SklearnClassificationModel:
    """Fit model according to parameters from train_params"""
    if train_params.model_type == "RandomForestClassifier":
        rf_params = train_params.classifier_params.to_dict()
        model = RandomForestClassifier(**rf_params)
    elif train_params.model_type == "LogisticRegression":
        lr_params = train_params.classifier_params.to_dict()
        model = LogisticRegression(**lr_params)
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict probabilities using previously fitted model"""
    predicts = model.predict_proba(features)[:, 1]
    return predicts


def get_predicted_labels(predicts: np.ndarray, threshold: float) -> np.ndarray:
    """Predict class labels according to probability threshold"""
    predicted_labels = (predicts > threshold).astype(int)
    return predicted_labels


def evaluate_model(
        predicts: np.ndarray,
        target: pd.Series,
        threshold: float = 0.5
) -> Dict[str, float]:
    """Estimate a model performance using the following metrics:
        - ROC AUC
        - F1 measure
        - Accuracy
    """
    predicted_labels = get_predicted_labels(predicts, threshold)
    calculated_metrics = {
        "roc_auc_score": roc_auc_score(target, predicts),
        "f1_score": f1_score(target, predicted_labels),
        "accuracy": accuracy_score(target, predicted_labels),
    }
    return calculated_metrics


def save_model(model: SklearnClassificationModel, output_filepath: str) -> str:
    """Save pickled model to file"""
    with open(output_filepath, "wb") as f:
        pickle.dump(model, f)
    return output_filepath


def load_model(model_filepath: str) -> SklearnClassificationModel:
    """Load pickled model from file"""
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    return model
