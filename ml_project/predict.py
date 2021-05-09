import datetime
import os
from typing import Tuple

import click
import numpy as np
import pandas as pd

from src.data.make_dataset import (
    download_data_from_s3,
    read_dataset
)
from src.features.build_features import (
    make_features,
    build_transformer
)
from src.logs.logger import setup_logger
from src.models.model_fit_predict import (
    load_model,
    predict_probabilities
)
from src.params.predict_pipeline_params import (
    read_predict_pipeline_params,
    PredictPipelineParams
)

logger = setup_logger("predict.py")


def save_predictions(filepath: str, predicted_probabilities: np.array) -> str:
    """Save predictions: np.array to .csv file"""
    tmp = pd.DataFrame({"target": predicted_probabilities})
    tmp.to_csv(filepath, index=False)
    return filepath


def predict_pipeline(params: PredictPipelineParams) -> Tuple[str, pd.DataFrame]:
    """Inference (predict) pipeline with the following steps:
        1. Optional: download dataset from remote server (S3)
        2. Read dataset to pandas.DataFrame
        3. Preprocess features
        4. Load trained model
        5. Make predictions
        6. Save predictions
    """
    start = datetime.datetime.now()
    logger.info("\nstart predict pipeline\n")

    download_params = params.download_params
    if download_params:
        logger.info(f"start loading dataset from s3://{download_params.s3_bucket}")
        os.makedirs(download_params.output_folder, exist_ok=True)
        dataset_filepath = download_data_from_s3(
            download_params.s3_bucket,
            download_params.s3_filepath,
            download_params.output_folder
        )
        logger.debug(f"dataset was successfully loaded to {download_params.output_folder}")
    else:
        dataset_filepath = params.input_data_path

    logger.info("start reading dataset to pandas.DataFrame")
    data = read_dataset(dataset_filepath)

    logger.info("start preprocessing")
    transformer = build_transformer(params.feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)

    logger.info("start loading trained model")
    model = load_model(params.model_path)

    logger.info("start making predictions")
    predicts = predict_probabilities(model, features)

    logger.info("start saving predictions")
    predictions_filepath = save_predictions(params.predictions_path, predicts)

    end = datetime.datetime.now()
    logger.info(f"\nprediction pipeline finished. Time: {end - start}\n")

    return predictions_filepath, predicts


@click.command(name="predict")
@click.argument("config_path")
def predict_command(config_path: str):
    """Command line interface for inference (predict) pipeline"""
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_command()
