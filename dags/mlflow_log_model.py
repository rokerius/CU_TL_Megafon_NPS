import mlflow
import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from airflow.decorators import task, dag
from datetime import datetime


# DAG для логирования модели в MLflow
@dag(start_date=datetime(2024, 1, 1), schedule="@daily", catchup=False)
def D_feature_engineering_pg_dag():
    @task
    def train_and_log_model():
        # Заглушка данных
        df = pd.DataFrame({
            "year": [2023, 2024, 2025],
            "month": [1, 2, 3],
            "nps": [50, 45, 60]
        })

        X = df[["year", "month"]]
        y = df["nps"]

        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        # MLflow настройки
        mlflow.set_tracking_uri("file:/tmp/mlruns")  # локально вместо S3
        mlflow.set_registry_uri("file:/tmp/mlruns")  # для модели

        mlflow.set_experiment("feature_engineering_pg")

        with mlflow.start_run(run_name="linear_regression_nps"):
            mlflow.log_param("features", ["year", "month"])
            mlflow.log_metric("mse", mse)

            # сохраняем модель во временный файл
            model_path = "/tmp/linreg.pkl"
            joblib.dump(model, model_path)

            # логируем артефакт (будет в /tmp/mlruns)
            mlflow.log_artifact(model_path)

        return mse

    train_and_log_model()


dag = D_feature_engineering_pg_dag()
