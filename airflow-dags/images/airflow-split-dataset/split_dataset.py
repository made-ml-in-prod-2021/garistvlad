import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split_dataset")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--val_size", default=0.2, help="share of validation dataset")
@click.option("--seed", default=42, help="random seed")
def split_dataset(input_dir: str, output_dir: str, val_size: float, seed: int):
    if not os.path.exists(os.path.join(input_dir, 'data.csv')):
        raise ValueError(f"There are no file 'data.csv' inside {input_dir}")
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    if not os.path.exists(os.path.join(input_dir, 'target.csv')):
        raise ValueError(f"There are no file 'target.csv' inside {input_dir}")
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), squeeze=True)

    data_train, data_val, target_train, target_val = train_test_split(
        data, target, train_size=val_size, shuffle=True, random_state=seed
    )

    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    data_train.to_csv(os.path.join(output_dir, "train", "data.csv"), header=True, index=False)
    target_train.to_csv(os.path.join(output_dir, "train", "target.csv"), header=True, index=False)

    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    data_train.to_csv(os.path.join(output_dir, "val", "data.csv"), header=True, index=False)
    target_train.to_csv(os.path.join(output_dir, "val", "target.csv"), header=True, index=False)


if __name__ == "__main__":
    split_dataset()
