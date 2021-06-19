from datetime import timedelta
import os
import sys

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.utils.dates import days_ago

sys.path.append(os.path.dirname(__file__))
from utils import load_default_args

DAG_NAME = "predict"
DAG_SHORT_DESCRIPTION = "Use fitted model to make predictions"
EXECUTION_DATE = "{{ ds }}"
AIRFLOW_BASE_DIR = "usr/local/airflow"
LOCAL_FS_DATA_DIR = "/Users/gari/Education/made/prod_ml/my_repo/airflow-dags/data"


with DAG(
    DAG_NAME,
    default_args=load_default_args(),
    description=DAG_SHORT_DESCRIPTION,
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
) as dag:

    task_start = DummyOperator(task_id="start")
    check_data = FileSensor(
        task_id="check-data-exists",
        filepath=f"{AIRFLOW_BASE_DIR}/data/raw/{EXECUTION_DATE}/data.csv",
        poke_interval=30,
        retries=100,
    )
    check_transformer = FileSensor(
        task_id="check-transformer-exists",
        filepath=f"{AIRFLOW_BASE_DIR}/data/models/transformer.pickle",
        poke_interval=30,
        retries=100,
    )
    check_model = FileSensor(
        task_id="check-model-exists",
        filepath=f"{AIRFLOW_BASE_DIR}/data/models/model.pickle",
        poke_interval=30,
        retries=100,
    )

    transform_command = \
        f"--input-dir='/data/raw/{EXECUTION_DATE}' " \
        f"--output-dir='/data/transformed/{EXECUTION_DATE}' " \
        f"--transformer-dir='/data/models' " \
        "--no-target"
    task_transform_data = DockerOperator(
        image="garistvlad/airflow-transform",
        command=transform_command,
        network_mode="bridge",
        task_id="docker-airflow-transform-data",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    predict_command = \
        f"--input-dir='/data/transformed/{EXECUTION_DATE}' " \
        f"--output-dir='/data/predictions/{EXECUTION_DATE}' " \
        f"--model-dir='/data/models'"
    task_predict = DockerOperator(
        image="garistvlad/airflow-predict",
        command=predict_command,
        network_mode="bridge",
        task_id="docker-airflow-predict",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    task_start >> [
        check_data, check_transformer, check_model
    ] >> task_transform_data >> task_predict
