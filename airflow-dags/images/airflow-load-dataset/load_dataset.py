"""Data generation process based on Diabetes dataset from sklearn"""
import os

import click
import numpy as np
from sklearn.datasets import load_diabetes


@click.command("load_dataset")
@click.argument("output_dir")
@click.option("--seed", default=42, help="random seed")
def load_dataset(output_dir: str, seed: int):
    """Load Diabetes dataset from sklearn, provide uncertainty and save"""
    np.random.seed(seed)
    data, target = load_diabetes(return_X_y=True, as_frame=True)
    data.loc[:, :] = data.values + np.random.random(data.shape)
    target = target + np.random.randint(-target.min() // 2, target.min() // 2, size=target.shape)

    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False, header=True)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    load_dataset()
