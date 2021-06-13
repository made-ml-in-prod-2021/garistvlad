import datetime
import json
import os

import click

from src.data.make_dataset import (
    download_data_from_s3,
    read_dataset,
    train_val_split
)
from src.features.build_features import (
    build_transformer,
    make_features,
    extract_target,
    fit_transformer
)
from src.logs.logger import setup_logger
from src.models.model_fit_predict import (
    train_model,
    save_model,
    predict_probabilities,
    evaluate_model
)
from src.params.train_pipeline_params import (
    TrainPipelineParams,
    read_train_pipeline_params
)

logger = setup_logger("train.py")


def train_pipeline(params: TrainPipelineParams):
    """Train pipeline with the following steps:
        1. Optional: download dataset from remote server (S3)
        2. Read dataset to pandas.DataFrame
        3. Split dataset on Train and Validation sets
        4. Preprocess features
        5. Train model using Train set
        6. Evaluate model and calculate metrics on Validation set
        7. Save metrics to file.json
        8. Save model to file.pickle

    Returns filepath to trained model and dist of metrics
    """
    start = datetime.datetime.now()
    logger.info("start training pipeline")

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
    logger.debug(f"data.shape = {data.shape}")

    logger.info("start splitting data into train and validation")
    train_df, val_df = train_val_split(data, params.split_params)
    logger.debug(f"train_df.shape = {train_df.shape}")
    logger.debug(f"val_df.shape = {val_df.shape}")

    logger.info("start preprocessing")
    transformer = build_transformer(params.feature_params)
    transformer, train = fit_transformer(transformer, train_df)
    train_features = make_features(transformer, train)
    train_target = extract_target(train, params.feature_params)
    logger.debug(f"train_features.shape = {train_features.shape}")

    logger.info("start training model")
    model = train_model(
        train_features,
        train_target,
        params.train_params
    )

    logger.info("start evaluating the model")
    val_features = make_features(transformer, val_df)
    val_target = extract_target(val_df, params.feature_params)
    logger.debug(f"val_features.shape = {val_features.shape}")
    probabilities = predict_probabilities(
        model,
        val_features
    )
    metrics = evaluate_model(
        probabilities,
        val_target,
    )
    logger.debug(f"calculated metrics: {metrics}")

    logger.info("start saving metrics to json")
    with open(params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)

    logger.info("start saving the model just trained")
    path_to_model = save_model(model, params.output_model_path)

    end = datetime.datetime.now()
    logger.info(f"training pipeline finished. Time: {end - start}\n")

    return path_to_model, metrics


@click.command(name="train")
@click.argument("config_path")
def train_command(config_path: str):
    """Command line interface for training pipeline"""
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_command()
