from datetime import timedelta
import os
import random
import sys

from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.utils.dates import days_ago

sys.path.append(os.path.dirname(__file__))
from utils import load_default_args

DAG_NAME = "load_dataset"
DAG_SHORT_DESCRIPTION = "Load dataset and save to output-dir"
EXECUTION_DATE = "{{ ds }}"
LOCAL_FS_DATA_DIR = "/Users/gari/Education/made/prod_ml/my_repo/airflow-dags/data"


with DAG(
    DAG_NAME,
    default_args=load_default_args(),
    description=DAG_SHORT_DESCRIPTION,
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
) as dag:
    task_start = DummyOperator(task_id="start")
    task_load = DockerOperator(
        image="garistvlad/airflow-load-dataset",
        command=f"/data/raw/{EXECUTION_DATE} --seed={random.randint(0, 1000)}",
        network_mode="bridge",
        task_id="docker-airflow-load-dataset",
        xcom_push=False,
        volumes=[f"{LOCAL_FS_DATA_DIR}:/data"]
    )

    task_start >> task_load
