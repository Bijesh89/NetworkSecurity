
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd

app = FastAPI()

MODEL_URI = "models:/RandomForest_CarPrice/Production"

model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message": "Car Price Prediction API"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }
