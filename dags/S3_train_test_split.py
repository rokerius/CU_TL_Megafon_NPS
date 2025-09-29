from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

# ===== Настройки =====
S3_BUCKET = "cu-mf-project"
S3_INPUT_KEY = "features/df_with_features_1.csv"  # исходный файл
S3_TRAIN_KEY = "features/train.csv"
S3_TEST_KEY = "features/test.csv"

LOCAL_INPUT_PATH = "/tmp/df_with_features.csv"
LOCAL_TRAIN_PATH = "/tmp/train.csv"
LOCAL_TEST_PATH = "/tmp/test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 239

# ===== Функции =====
def get_s3_client():
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

def download_from_s3(**kwargs):
    s3 = get_s3_client()
    Path(LOCAL_INPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(S3_BUCKET, S3_INPUT_KEY, LOCAL_INPUT_PATH)
    print(f"Downloaded s3://{S3_BUCKET}/{S3_INPUT_KEY} -> {LOCAL_INPUT_PATH}")

def split_data(**kwargs):
    df = pd.read_csv(LOCAL_INPUT_PATH)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df.to_csv(LOCAL_TRAIN_PATH, index=False)
    test_df.to_csv(LOCAL_TEST_PATH, index=False)
    print(f"Train/Test split done. Train: {LOCAL_TRAIN_PATH}, Test: {LOCAL_TEST_PATH}")

def upload_to_s3(local_path, s3_key):
    s3 = get_s3_client()
    s3.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"Uploaded {local_path} -> s3://{S3_BUCKET}/{s3_key}")

def upload_train(**kwargs):
    upload_to_s3(LOCAL_TRAIN_PATH, S3_TRAIN_KEY)

def upload_test(**kwargs):
    upload_to_s3(LOCAL_TEST_PATH, S3_TEST_KEY)

# ===== DAG =====
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

with DAG(
    dag_id='s3_train_test_split',
    description='DAG, который скачивает данные из S3, делит на train/test и загружает обратно в S3',
    default_args=default_args,
    catchup=False,
) as dag:

    t1_download = PythonOperator(
        task_id='download_from_s3',
        python_callable=download_from_s3
    )

    t2_split = PythonOperator(
        task_id='split_data',
        python_callable=split_data
    )

    t3_upload_train = PythonOperator(
        task_id='upload_train',
        python_callable=upload_train
    )

    t4_upload_test = PythonOperator(
        task_id='upload_test',
        python_callable=upload_test
    )

    # ===== Порядок выполнения =====
    t1_download >> t2_split >> [t3_upload_train, t4_upload_test]
