import os

import pandas as pd

from train import train_pipeline
from predict import predict_pipeline
from src.params.split_params import SplitParams
from src.params.feature_params import FeatureParams
from src.params.train_params import TrainParams, LogisticRegressionParams
from src.params.train_pipeline_params import TrainPipelineParams
from src.params.predict_pipeline_params import PredictPipelineParams

from py._path.local import LocalPath


def test_train_e2e(
        tmpdir: LocalPath,
        sample_data: pd.DataFrame,
        feature_params: FeatureParams
):
    data_path = tmpdir.join('sample.csv')
    sample_data.to_csv(data_path)

    expected_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")

    path_to_model, metrics = train_pipeline(
        params=TrainPipelineParams(
            input_data_path=data_path,
            output_model_path=expected_model_path,
            metric_path=expected_metric_path,
            download_params=None,
            split_params=SplitParams,
            feature_params=feature_params,
            train_params=TrainParams(
                model_type="LogisticRegression",
                classifier_params=LogisticRegressionParams(
                    C=1,
                    max_iter=50,
                    random_state=101,
                ),
            )
        )
    )

    assert "roc_auc_score" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics
    assert path_to_model == expected_model_path
    assert os.path.exists(path_to_model)


def test_predict_e2e(
        tmpdir: LocalPath,
        sample_data: pd.DataFrame,
        feature_params: FeatureParams
):
    data_path = tmpdir.join('sample.csv')
    sample_data.to_csv(data_path)
    model_path = tmpdir.join("model.pkl")
    metric_path = tmpdir.join("metrics.json")
    # Fit
    path_to_model, metrics = train_pipeline(
        params=TrainPipelineParams(
            input_data_path=data_path,
            output_model_path=model_path,
            metric_path=metric_path,
            download_params=None,
            split_params=SplitParams,
            feature_params=feature_params,
            train_params=TrainParams(
                model_type="LogisticRegression",
                classifier_params=LogisticRegressionParams(
                    C=1,
                    max_iter=50,
                    random_state=101,
                ),
            )
        )
    )
    # Predict
    expected_predictions_path = tmpdir.join("predictions.csv")
    predictions_filepath, predicts = predict_pipeline(
        PredictPipelineParams(
            input_data_path=data_path,
            model_path=model_path,
            predictions_path=expected_predictions_path,
            download_params=None,
            feature_params=feature_params,
        )
    )
    assert expected_predictions_path == predictions_filepath
    assert len(predicts) == sample_data.shape[0]
