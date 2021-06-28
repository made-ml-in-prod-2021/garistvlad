import datetime
import os
import time
import shutil
from typing import Optional

from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from .custom_datamodels import (
    HeartRequestModel,
    HeartResponseModel
)
from .utils.inference import predict_pipeline
from .utils.loader import load_pickled
from .utils.logs import setup_logger


BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.path.pardir
    )
)  # .../api
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, "app", "pretrained_models")
TMP_DATA_DIR = os.path.join(BASE_DIR, "app", "tmp_data")

START_DELAY = 30
CRASH_DELAY = 180

logger = setup_logger(name="main:app")
transformer: Optional[ColumnTransformer] = None
classifier: Optional[LogisticRegression] = None

app = FastAPI()
start = datetime.datetime.now()


@app.on_event("startup")
def provide_delayed_start():
    """Start app after `START_DELAY` seconds"""
    time.sleep(START_DELAY)


@app.on_event("startup")
def load_pretrained_transformer():
    """Load transformer previously fitted"""
    global transformer
    filepath = os.path.join(PRETRAINED_MODELS_DIR, "transformer.pickle")
    try:
        loaded_transformer = load_pickled(filepath)
    except FileNotFoundError as err:
        logger.error(err)
        return
    transformer = loaded_transformer


@app.on_event("startup")
def load_pretrained_classifier():
    """Load classifier previously fitted"""
    global classifier
    filepath = os.path.join(PRETRAINED_MODELS_DIR, "classifier.pickle")
    try:
        loaded_classifier = load_pickled(filepath)
    except FileNotFoundError as err:
        logger.error(err)
        return
    classifier = loaded_classifier


@app.on_event("startup")
def load_real_dataset():
    """Load real dataset from S3 and copy locally to tmp directory"""
    s3_filepath = os.environ.get("HEART_DISEASE_S3_URL")
    if not s3_filepath:
        logger.error("There are no s3 url with real dataset specified")
        return
    tmp_df = pd.read_csv(s3_filepath)
    if not os.path.exists(TMP_DATA_DIR):
        os.mkdir(TMP_DATA_DIR)
    tmp_df.to_csv(f"{TMP_DATA_DIR}/real_dataset.csv", header=True, index=False)


@app.on_event("shutdown")
def remove_tmp_dir():
    """Remove tmp directory on shutdown"""
    if os.path.exists(TMP_DATA_DIR):
        shutil.rmtree(TMP_DATA_DIR)


@app.get("/")
async def root():
    """Just for seeing not empty root-page"""
    description = "MADE. ML in production. HA #2: REST service"
    return {"description": description}


@app.get("/health")
def check_health() -> bool:
    """Check: pretrained classifier and transformer loaded correctly"""
    if (datetime.datetime.now() - start).seconds > CRASH_DELAY:
        raise OSError("App crashed as expected")

    transformer_is_loaded = (transformer is not None)
    classifier_is_loaded = (classifier is not None)
    pretrained_loaded = (classifier_is_loaded and transformer_is_loaded)
    return pretrained_loaded


@app.post("/predict", response_model=HeartResponseModel)
def make_prediction(data: HeartRequestModel) -> HeartResponseModel:
    if not check_health():
        logger.error("Model is not loaded")
        raise HTTPException(
            status_code=500,
            detail="Model should be loaded for making predictions"
        )
    predicted_probability = predict_pipeline(
        data=data.to_pandas(),
        transformer=transformer,
        classifier=classifier
    )
    return {"predicted_probability": predicted_probability}
