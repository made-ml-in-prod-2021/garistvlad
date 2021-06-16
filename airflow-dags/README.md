# Home Assignment #3
## Model training and usage on schedule with Apache Airflow

How to run airflow:
```cmd
# 0. Create .env file inside `airflow-dags` dir with the following parameters for getting e-mail alerts:
SMTP_USER=...
SMTP_PASSWORD=...

# 1. Build services:
docker-compose up -d --build

# 2. Check that everything works correctly without errors:
docker-compose logs
  ...
  webserver_1        | [2021-06-04 20:42:37,174] {{__init__.py:51}} INFO - Using executor LocalExecutor
  webserver_1        | [2021-06-04 20:42:37,174] {{dagbag.py:403}} INFO - Filling up the DagBag from /usr/local/airflow/dags
  webserver_1        | [2021-06-04 20:42:37 +0000] [174] [INFO] Handling signal: ttou
  webserver_1        | [2021-06-04 20:42:37 +0000] [58430] [INFO] Worker exiting (pid: 58430)
  ...
```

Web-interface can be found at: http://localhost:5050/

There are the following DAGs described in `/dags` folder:
- `load_dataset` that generates data on a daily basis
- `train_model` that fitted transformer and regression model on a weekly basis
- `predict` that provide model inference on a daily basis
