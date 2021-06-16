import os
import random
import sys

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.utils.dates import days_ago

sys.path.append(os.path.dirname(__file__))
from utils import load_default_args

DAG_NAME = "train_model"
DAG_SHORT_DESCRIPTION = "Transform data and fit regression model"
EXECUTION_DATE = "{{ ds }}"
AIRFLOW_BASE_DIR = "usr/local/airflow"
LOCAL_FS_DATA_DIR = "/Users/gari/Education/made/prod_ml/my_repo/airflow-dags/data"


with DAG(
    DAG_NAME,
    default_args=load_default_args(),
    description=DAG_SHORT_DESCRIPTION,
    schedule_interval="0 5 * * 6",  # weekly, every Saturday at 05:00
    start_date=days_ago(7),
) as dag:

    task_start = DummyOperator(task_id="start")
    check_data = FileSensor(
        task_id="check-data-exists",
        filepath=f"{AIRFLOW_BASE_DIR}/data/raw/{EXECUTION_DATE}/data.csv",
        poke_interval=30,
        retries=50,
    )
    check_target = FileSensor(
        task_id="check-target-exists",
        filepath=f"{AIRFLOW_BASE_DIR}/data/raw/{EXECUTION_DATE}/target.csv",
        poke_interval=30,
        retries=50,
    )

    raw_data_split_command = \
        f"--input-dir='/data/raw/{EXECUTION_DATE}' " \
        f"--output-dir='/data/raw-split/{EXECUTION_DATE}' " \
        f"--val_size=0.2 " \
        f"--seed={random.randint(0, 1000)}"
    task_split_raw_data = DockerOperator(
        image="garistvlad/airflow-split-dataset",
        command=raw_data_split_command,
        network_mode="bridge",
        task_id="docker-airflow-split-raw-dataset",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    fit_transformer_command = \
        f"--input-dir='/data/raw-split/{EXECUTION_DATE}/train' " \
        f"--output-dir='/data/models'"
    task_fit_transformer = DockerOperator(
        image="garistvlad/airflow-fit-transformer",
        command=fit_transformer_command,
        network_mode="bridge",
        task_id="docker-airflow-fit-transformer",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    transform_command = \
        f"--input-dir='/data/raw-split/{EXECUTION_DATE}' " \
        f"--output-dir='/data/transformed-split/{EXECUTION_DATE}' " \
        f"--transformer-dir='/data/models'"
    task_transform_data = DockerOperator(
        image="garistvlad/airflow-transform",
        command=transform_command,
        network_mode="bridge",
        task_id="docker-airflow-transform-data",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    fit_command = \
        f"--input-dir='/data/transformed-split/{EXECUTION_DATE}/train' " \
        f"--output-dir='/data/models'"
    task_fit_model = DockerOperator(
        image="garistvlad/airflow-fit-model",
        command=fit_command,
        network_mode="bridge",
        task_id="docker-airflow-fit-model",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    validate_command = \
        f"--input-dir='/data/transformed-split/{EXECUTION_DATE}/val' " \
        f"--output-dir='/data/metrics' " \
        f"--model-dir='/data/models'"
    task_validate_model = DockerOperator(
        image="garistvlad/airflow-validate",
        command=validate_command,
        network_mode="bridge",
        task_id="docker-airflow-validate-model",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    task_start >> [check_data, check_target] >> task_split_raw_data >> \
        task_fit_transformer >> task_transform_data >> task_fit_model >> task_validate_model
