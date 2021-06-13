from dataclasses import dataclass
from typing import Optional

from marshmallow_dataclass import class_schema
import yaml

from .download_params import DownloadParams
from .split_params import SplitParams
from .feature_params import FeatureParams
from .train_params import TrainParams


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    download_params: Optional[DownloadParams]
    split_params: SplitParams
    feature_params: FeatureParams
    train_params: TrainParams


TrainingPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    """Load config.yaml into dataclass with target params structure"""
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
