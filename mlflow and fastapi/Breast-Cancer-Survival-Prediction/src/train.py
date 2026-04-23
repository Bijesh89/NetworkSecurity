from src.ingest_data import ingest_data
from src.preprocessing import data_preprocessing
from src.hyperparameters import search_hyperparameters

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
import pickle
import os

def train_model():
    # Load + preprocess
    data = ingest_data()
    X_train, X_test, y_train, y_test = data_preprocessing(data)

    # Hyperparameters (for CatBoost or extend later)
    params = search_hyperparameters()

    # Define models in SAME format
    models = {
        "catboost": CatBoostClassifier(**params, silent=True),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    os.makedirs("model", exist_ok=True)

    # MLflow experiment
    mlflow.set_tracking_uri("http://localhost:5000")  # Adjust if needed
    mlflow.set_experiment("classification_experiments")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Log params (if available)
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Log model
            mlflow.sklearn.log_model(model, name)

            # Save locally (optional)
            with open(f"model/{name}.pkl", "wb") as f:
                pickle.dump(model, f)

            print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")