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

def get_s3_client():
    """Создаём клиент S3 для Yandex Object Storage."""
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

# ===== S3 Helper =====
def download_from_s3(s3_bucket: str, s3_key: str, local_path: str):
    s3 = get_s3_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(s3_bucket, s3_key, local_path)
    print(f"Downloaded s3://{s3_bucket}/{s3_key} -> {local_path}")

def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    print(f"Uploaded {local_path} -> s3://{s3_bucket}/{s3_key}")

# ===== Основной процесс =====
download_from_s3(S3_BUCKET, S3_INPUT_KEY, LOCAL_INPUT_PATH)

df = pd.read_csv(LOCAL_INPUT_PATH)

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_df.to_csv(LOCAL_TRAIN_PATH, index=False)
test_df.to_csv(LOCAL_TEST_PATH, index=False)

upload_to_s3(LOCAL_TRAIN_PATH, S3_BUCKET, S3_TRAIN_KEY)
upload_to_s3(LOCAL_TEST_PATH, S3_BUCKET, S3_TEST_KEY)
