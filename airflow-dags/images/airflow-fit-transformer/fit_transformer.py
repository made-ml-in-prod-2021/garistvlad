import os
import pickle

import click
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLS = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


def fit_transformer(data: pd.DataFrame) -> pd.DataFrame:
    """Process features of dataset and save transformer"""
    if not all([col in data.columns for col in FEATURE_COLS]):
        raise ValueError(f"Not all columns from {FEATURE_COLS} are in dataset")
    mm_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data[FEATURE_COLS])
    return mm_scaler


def save_transformer(transformer: object, filepath: str):
    """Save binary object to filepath"""
    with open(filepath, "wb") as f:
        pickle.dump(transformer, f, pickle.HIGHEST_PROTOCOL)


@click.command("fit_transformer")
@click.option("--input-dir")
@click.option("--output-dir")
def fit_and_save_transformer(input_dir: str, output_dir: str):
    if not os.path.exists(os.path.join(input_dir, "data.csv")):
        raise ValueError(f"There are no file 'data.csv' inside {input_dir}")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    fitted_transformer = fit_transformer(data)

    os.makedirs(output_dir, exist_ok=True)
    save_transformer(
        fitted_transformer,
        os.path.join(output_dir, "transformer.pickle")
    )


if __name__ == "__main__":
    fit_and_save_transformer()
