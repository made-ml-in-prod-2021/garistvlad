import os
import pickle

import click
import pandas as pd


def load_model(filepath: str):
    """Save binary object to filepath"""
    if not os.path.exists(filepath):
        raise ValueError(f"There are no such file {filepath}")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir: str, model_dir: str):
    if not os.path.exists(os.path.join(input_dir, "data.csv")):
        raise ValueError(f"There are no file 'data.csv' inside {input_dir}")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    if not os.path.exists(os.path.join(model_dir, "model.pickle")):
        raise ValueError(f"There are no file 'model.pickle' inside {model_dir}")
    model = load_model(os.path.join(model_dir, "model.pickle"))

    y_pred = model.predict(data)
    y_pred = pd.Series(y_pred, index=data.index, name='target')
    os.makedirs(output_dir, exist_ok=True)
    y_pred.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()
