from typing import Dict, Union
import os
import comet_ml
from sklearn.metrics import f1_score
#import catboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from comet_ml import Experiment
import optuna
import pandas as pd
import numpy as np

from src.ingest_data import ingest_data
from src.preprocessing import data_preprocessing
from src.logger import get_console_logger

logger = get_console_logger('Hyperparameters Tuning')


def objective(trial):

    # Select model
    model_name = trial.suggest_categorical(
        "model", ["catboost", "logreg", "rf", "xgb"]
    )

    data = ingest_data()
    X_train, X_test, y_train, y_test = data_preprocessing(data)

    if model_name == "catboost":
        params = {
            "learning_rate": trial.suggest_float('learning_rate', 0.001, 0.2),
            "iterations": trial.suggest_int('iterations', 100, 1100),
            "depth": trial.suggest_int('depth', 3, 10),
            "loss_function": "MultiClass",
            "silent": True
        }
        model = CatBoostClassifier(**params)

    elif model_name == "logreg":
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        }
        model = LogisticRegression(**params)

    elif model_name == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
        }
        model = RandomForestClassifier(**params)

    elif model_name == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eval_metric": "mlogloss"
        }
        model = XGBClassifier(**params)

    # Train + evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = f1_score(y_test, y_pred, average='weighted')

    return score

def search_hyperparameters()-> Dict:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective,n_trials=20)
    best_params = study.best_params
    best_value = study.best_value
    
    experiment = Experiment(
        api_key="qaUy62jElVin2dR5B7isdybJF",
        project_name="Brest Cancer Survival Prediction",
    )
    Experiment(api_key="qaUy62jElVin2dR5B7isdybJF",auto_output_logging="default")
    # split best_params into preprocessing and model hyper-parameters
    best_preprocessing_hyperparams = {key: value for key, value in best_params.items() if key.startswith('pp_')}
    
    best_model_hyperparams = {
        key: value for key, value in best_params.items() if not key.startswith('pp_')}

    logger.info("Best Parameters:")
    for key, value in best_params.items():
        logger.info(f"{key}: {value}")
    logger.info(f"Best brier score: {best_value}")

    experiment.log_metric('Cross_validation_MAE', best_value)

    return best_preprocessing_hyperparams
    #return study.best_params