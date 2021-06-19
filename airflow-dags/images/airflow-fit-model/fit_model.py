import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def get_regression_model():
    """Define regression model for further fitting"""
    model_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=7,
        max_features=0.6,
        bootstrap=True,
        n_jobs=-1,
        max_samples=0.9
    )
    return model_rf


def save_model(model: object, filepath: str):
    """Save binary object to filepath"""
    with open(filepath, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)


@click.command("fit_model")
@click.option("--input-dir")
@click.option("--output-dir")
def fit_and_save_model(input_dir: str, output_dir: str):
    if not os.path.exists(os.path.join(input_dir, "data.csv")):
        raise ValueError(f"There are no file 'data.csv' inside {input_dir}")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    if not os.path.exists(os.path.join(input_dir, "target.csv")):
        raise ValueError(f"There are no file 'target.csv' inside {input_dir}")
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), squeeze=True)

    model_rf = get_regression_model()
    model_rf.fit(data, target)

    os.makedirs(output_dir, exist_ok=True)
    save_model(
        model_rf,
        os.path.join(output_dir, "model.pickle")
    )


if __name__ == "__main__":
    fit_and_save_model()
