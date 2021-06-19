import os
import pickle

import click
import pandas as pd


def load_transformer(filepath: str):
    """Save binary object to filepath"""
    if not os.path.exists(filepath):
        raise ValueError(f"There are no such file {filepath}")
    with open(filepath, "rb") as f:
        transformer = pickle.load(f)
    return transformer


@click.command("transform")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--transformer-dir")
@click.option("--no-target", is_flag=True)
def transform(input_dir: str, output_dir: str, transformer_dir: str, no_target: bool):
    if not os.path.exists(os.path.join(transformer_dir, "transformer.pickle")):
        raise ValueError(f"File 'transformer.pickle' not found inside {transformer_dir}")
    transformer = load_transformer(os.path.join(transformer_dir, "transformer.pickle"))

    if all([i in os.listdir(input_dir) for i in ['train', 'val']]):
        for sub_dir in ['train', 'val']:
            data = pd.read_csv(os.path.join(input_dir, sub_dir, "data.csv"))
            data_transformed = pd.DataFrame(
                transformer.transform(data),
                columns=data.columns,
                index=data.index
            )

            os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
            data_transformed.to_csv(
                os.path.join(output_dir, sub_dir, "data.csv"),
                index=False,
                header=True
            )
            if not no_target:
                target = pd.read_csv(os.path.join(input_dir, sub_dir, "target.csv"), squeeze=True)
                target.to_csv(os.path.join(output_dir, sub_dir, "target.csv"), index=False)
    else:
        data = pd.read_csv(os.path.join(input_dir, "data.csv"))
        data_transformed = pd.DataFrame(
            transformer.transform(data),
            columns=data.columns,
            index=data.index
        )
        os.makedirs(output_dir, exist_ok=True)
        data_transformed.to_csv(
            os.path.join(output_dir, "data.csv"),
            index=False,
            header=True
        )

        if not no_target:
            target = pd.read_csv(os.path.join(input_dir, "target.csv"), squeeze=True)
            target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    transform()
