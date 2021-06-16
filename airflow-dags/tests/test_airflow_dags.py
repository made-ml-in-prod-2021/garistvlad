import os

import pytest
from airflow.models import DagBag


AIRFLOW_BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)  # ../


@pytest.fixture()
def dag_bag():
    return DagBag(os.path.join(AIRFLOW_BASE_DIR, "dags"), include_examples=False)


def test_dags_are_loaded(dag_bag):
    assert not dag_bag.import_errors, (
        f"Dags loaded with errors: {dag_bag.import_errors}"
    )
    assert all(
        [dag_name in dag_bag.dags for dag_name in ['load_dataset', 'predict', 'train_model']]
    ), "Not all relevant dags were presented"


@pytest.mark.parametrize(
    ["dag_name", "expected_number_of_tasks", "task_list"],
    [
        pytest.param("load_dataset", 2, [
            'start',
            'docker-airflow-load-dataset'
        ]),
        pytest.param("train_model", 8, [
            'start',
            'check-data-exists',
            'check-target-exists',
            'docker-airflow-split-raw-dataset',
            'docker-airflow-fit-transformer',
            'docker-airflow-transform-data',
            'docker-airflow-fit-model',
            'docker-airflow-validate-model',
        ]),
        pytest.param("predict", 6, [
            'start',
            'check-data-exists',
            'check-transformer-exists',
            'check-model-exists',
            'docker-airflow-transform-data',
            'docker-airflow-predict',
        ]),
    ]
)
def test_dag_contains_all_appropriate_tasks(
        dag_name, expected_number_of_tasks, task_list, dag_bag
):
    dag = dag_bag.dags.get(dag_name)
    assert len(dag.tasks) == expected_number_of_tasks, (
        f"Wrong number of tasks: {len(dag.tasks)} instead of {expected_number_of_tasks}"
    )
    for i in range(expected_number_of_tasks):
        assert dag.tasks[i].task_id == task_list[i], (
            f"Expected task at position #{i}: {task_list[i]}. "
            f"But your task: {dag.tasks[i].task_id}"
        )
