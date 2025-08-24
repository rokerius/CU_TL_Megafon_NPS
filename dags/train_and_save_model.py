from datetime import datetime
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3


S3_BUCKET = "cu-mf-project"
FEATURES_PATH = f"s3://{S3_BUCKET}/features/df_with_features_1.csv"
LOCAL_FEATURES_PATH = "/tmp/df_with_features.csv"
MODEL_OUTPUT_PATH = "data/model/random_forest.pkl"
S3_MODEL_KEY = "models/random_forest.pkl"   # <- куда положим модель в S3


def get_s3_client():
    """Создаём клиент S3 для Yandex Object Storage."""
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def download_from_s3():
    """Скачиваем csv с признаками из S3."""
    s3 = get_s3_client()
    bucket = S3_BUCKET
    key = "features/df_with_features_1.csv"

    os.makedirs(os.path.dirname(LOCAL_FEATURES_PATH), exist_ok=True)
    s3.download_file(bucket, key, LOCAL_FEATURES_PATH)


def train_and_save_model():
    """Обучаем RandomForest, сохраняем модель локально и в S3."""
    df = pd.read_csv(LOCAL_FEATURES_PATH)

    feature_cols = [col for col in df.columns if col not in ["start_date", "year", "month", "nps", "state"]]
    X = df[feature_cols]
    y = df["nps"]

    model = RandomForestRegressor(n_estimators=100, random_state=239)

    model.fit(X, y)

    # Сохраняем локально
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    # Загружаем в S3
    s3 = get_s3_client()
    s3.upload_file(MODEL_OUTPUT_PATH, S3_BUCKET, S3_MODEL_KEY)
    print(f"Модель сохранена в S3: s3://{S3_BUCKET}/{S3_MODEL_KEY}")


with DAG(
    dag_id="train_and_save_random_forest",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["ml", "random_forest"],
) as dag:

    t1 = PythonOperator(
        task_id="download_features",
        python_callable=download_from_s3
    )

    t2 = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_and_save_model
    )

    t1 >> t2
