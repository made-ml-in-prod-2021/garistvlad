import json
import os
import pickle

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def load_model(filepath: str):
    """Save binary object to filepath"""
    if not os.path.exists(filepath):
        raise ValueError(f"There are no such file {filepath}")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


def save_metrics(metrics: object, filepath: str):
    """Save metrics to filepath"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f)


def calculate_metrics(y_true, y_pred):
    """Calculate metrics: MAPE, RMSE, MAE and R2"""
    metrics = {
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    return metrics


@click.command("validate_model")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def calculate_and_save_metrics(input_dir: str, output_dir: str, model_dir: str):
    if not os.path.exists(os.path.join(input_dir, "data.csv")):
        raise ValueError(f"There are no file 'data.csv' inside {input_dir}")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    if not os.path.exists(os.path.join(input_dir, "target.csv")):
        raise ValueError(f"There are no file 'target.csv' inside {input_dir}")
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), squeeze=True)

    if not os.path.exists(os.path.join(model_dir, "model.pickle")):
        raise ValueError(f"There are no file 'model.pickle' inside {model_dir}")
    model = load_model(os.path.join(model_dir, "model.pickle"))

    y_val = model.predict(data)
    validation_metrics = calculate_metrics(target, y_val)

    os.makedirs(output_dir, exist_ok=True)
    save_metrics(
        validation_metrics,
        os.path.join(output_dir, "metrics.json")
    )


if __name__ == "__main__":
    calculate_and_save_metrics()
