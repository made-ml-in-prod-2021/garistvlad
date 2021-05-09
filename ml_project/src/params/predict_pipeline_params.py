from typing import Optional
from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml

from .download_params import DownloadParams
from .feature_params import FeatureParams


@dataclass()
class PredictPipelineParams:
    input_data_path: str
    model_path: str
    predictions_path: str
    download_params: Optional[DownloadParams]
    feature_params: FeatureParams


PredictParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config_path: str) -> PredictPipelineParams:
    """Load config.yaml into dataclass with target params structure"""
    with open(config_path, 'r') as config:
        schema = PredictParamsSchema()
        return schema.load(yaml.safe_load(config))
