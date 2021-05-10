Home Assignment #1
==============================

**Short project description:**

MADE, ML in production course, HA #1: ML project base structure

**Initial setup:**
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add AWS credentials to .env file in root directory
AWS_ACCESS_KEY_ID = ...
AWS_SECRET_ACCESS_KEY = ...
```

**Train:**
```
# for RamdomForest classifier:
python train.py configs/train_rf_config.yaml

# for LogisticRegression classifier:
python train.py configs/train_lr_config.yaml
```

**Predict:**
```
python predict.py configs/predict_config.yaml
```

**Test:**
```
pytest tests/
```

Project Organization
------------

    ├── LICENSE
    ├── .gitignore
    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── train.py           <- Main module for training classifier
    ├── predict.py         <- Main module for making an inferense for classifier just trained
    |
    ├── data
    │   ├── predictions    <- The final predictions, obtained after training and model inference
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions and their metrics
    │
    ├── configs            <- Contains .yaml configuration files for model training and inference
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-username-initial-data-exploration.ipynb`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── __init__.py
    │   │   └── model_fit_preduct.py
    |   |
    │   ├── logs          <- Scripts to setup custom loggers
    │   │   ├── __init__.py
    │   │   └── logger.py
    │   │
    │   └── params        <- Scripts to create dataclasses with training and prediction parameters
    │       ├── __init__.py
    │       ├── download_params.py
    │       ├── feature_params.py
    │       ├── predict_pipeline_params.py
    │       ├── split_params.py
    │       ├── train_params.py
    │       └── train_pipeline_params.py
    │
    └── tests              <- Contains module and integration test cases


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
