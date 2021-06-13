from typing import Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def predict_pipeline(
        data: pd.DataFrame,
        transformer: ColumnTransformer,
        classifier: Union[LogisticRegression, RandomForestClassifier]
) -> float:
    """Transform initial data and make a prediction"""
    data_processed = transformer.transform(data)
    predicted_probability = classifier.predict_proba(data_processed)[0, 1]
    return predicted_probability
