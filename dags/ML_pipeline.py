from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

import pandas as pd
import boto3
import mlflow
import mlflow.sklearn

from pathlib import Path
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== Defaults =====
S3_BUCKET = Variable.get("s3_bucket", "cu-mf-project")
TRAIN_PATH = Variable.get("train_path", f"s3://{S3_BUCKET}/features/train.csv")
TEST_PATH = Variable.get("test_path", f"s3://{S3_BUCKET}/features/test.csv")
MODEL_PATH = Variable.get("s3_model_key", "models/random_forest.pkl")

LOCAL_TRAIN_PATH = Variable.get("local_train_path", "/tmp/train.csv")
LOCAL_TEST_PATH = Variable.get("local_test_path", "/tmp/test.csv")
LOCAL_MODEL_PATH = Variable.get("local_model_path", "/tmp/model.pkl")

MLFLOW_EXPERIMENT = Variable.get("mlflow_experiment_name", "default")
MLFLOW_S3_ENDPOINT = Variable.get("mlflow_s3_endpoint", "https://storage.yandexcloud.net")

# ===== DAG =====
default_args = {"owner": "airflow", "depends_on_past": False}

dag = DAG(
    "model_evaluation_regression",
    default_args=default_args,
    description="DAG, который берет твою модель из S3, делает предсказания на тестовом наборе и логирует результаты в MLflow",
    catchup=False,
)

# ===== S3 Client =====
def get_s3_client():
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url=MLFLOW_S3_ENDPOINT,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

def download_from_s3(s3_uri: str, local_path: str):
    s3 = get_s3_client()
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded {s3_uri} to {local_path}")


# ===== Tasks =====
@task(dag=dag)
def download_datasets():
    download_from_s3(TRAIN_PATH, LOCAL_TRAIN_PATH)
    download_from_s3(TEST_PATH, LOCAL_TEST_PATH)
    return {"train": LOCAL_TRAIN_PATH, "test": LOCAL_TEST_PATH}


@task(dag=dag)
def load_model():
    download_from_s3(f"s3://{S3_BUCKET}/{MODEL_PATH}", LOCAL_MODEL_PATH)
    print(f"Model downloaded to {LOCAL_MODEL_PATH}")
    return LOCAL_MODEL_PATH


@task(dag=dag)
def predict_and_log(paths: dict, model_path: str):
    model = joblib.load(model_path)

    results = {}
    datasets_info = {}
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT

    with mlflow.start_run():
        for split_name, path in paths.items():
            df = pd.read_csv(path)

            feature_cols = [col for col in df.columns if col not in ["start_date", "year", "month", "nps", "state"]]
            X = df[feature_cols]
            y = df["nps"]

            y_pred = model.predict(X)

            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            print(f"[{split_name.upper()}] RMSE: {rmse}, MAE: {mae}, R2: {r2}")
            results[split_name] = {"rmse": rmse, "mae": mae, "r2": r2}
            datasets_info[split_name] = {"rows": len(df), "features": len(feature_cols)}

            mlflow.log_metric(f"{split_name}_rmse", rmse)
            mlflow.log_metric(f"{split_name}_mae", mae)
            mlflow.log_metric(f"{split_name}_r2", r2)

            mlflow.log_param(f"{split_name}_rows", len(df))
            mlflow.log_param(f"{split_name}_features", len(feature_cols))

            sample_preds = pd.DataFrame({
                "y_true": y[:10].values,
                "y_pred": y_pred[:10]
            })
            mlflow.log_text(
                sample_preds.to_csv(index=False),
                artifact_file=f"{split_name}_examples/sample_predictions.csv"
            )

            mlflow.log_text(
                df.to_csv(index=False),
                artifact_file=f"{split_name}_data/{split_name}_dataset.csv"
            )

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(model, artifact_path="model")

    return results


# ===== Dependencies =====
datasets = download_datasets()
model_path = load_model()
predict_and_log(datasets, model_path)