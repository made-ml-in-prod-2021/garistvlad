import os
from typing import Tuple

from boto3 import client
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.params.split_params import SplitParams

PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
DOTENV_PATH = os.path.join(PROJECT_DIR, '.env')
load_dotenv(DOTENV_PATH)


def download_data_from_s3(
        s3_bucket: str,
        s3_filepath: str,
        output_folder: str
) -> str:
    """Read data from s3 AWS bucket and save locally to `output_folder`.
    Returns output filepath in the format: {output_folder}/{s3_filename}
    Access provided via secret_keys hidden from outside.
    """
    s3 = client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    output_filepath = os.path.join(output_folder, Path(s3_filepath).name)
    s3.download_file(
        s3_bucket,
        s3_filepath,
        output_filepath
    )
    return output_filepath


def read_dataset(csv_filepath: str) -> pd.DataFrame:
    """Read .csv file and return pandas.DataFrame"""
    df = pd.read_csv(csv_filepath)
    return df


def train_val_split(
        data: pd.DataFrame,
        params: SplitParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data on Train and Validation sets according to custom params"""
    if not 0 < params.val_size < 1:
        raise ValueError(
            "Validation size should be strictly greater than 0 "
            "and strictly less than 1"
        )
    train_data, val_data = train_test_split(
        data,
        test_size=params.val_size,
        shuffle=params.shuffle,
        random_state=params.random_state
    )
    return train_data, val_data
