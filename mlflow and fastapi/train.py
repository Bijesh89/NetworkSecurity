
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("./data/cardekho_imputated.csv", index_col=0)

# Preprocessing (from notebook)
df = df.dropna()

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("RandomForest_CarPrice")

with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

    print("RMSE:", rmse)
    print("R2:", r2)
