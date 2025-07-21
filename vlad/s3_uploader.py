# s3_uploader.py
import os
import boto3
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения из кастомного .env
load_dotenv(dotenv_path="config/megafon-access.env")

def upload_to_s3(local_path: str, s3_key: str):
    aws_access_key = os.getenv("S3_ACCESS_KEY")
    aws_secret_key = os.getenv("S3_SECRET_KEY")
    s3_endpoint = os.getenv("S3_ENDPOINT")
    s3_bucket = os.getenv("S3_BUCKET")

    if not all([aws_access_key, aws_secret_key, s3_endpoint, s3_bucket]):
        raise ValueError("❌ Не все переменные окружения заданы в .env")

    s3 = boto3.client(
        service_name='s3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    try:
        with open(local_path, "rb") as f:
            s3.upload_fileobj(f, s3_bucket, s3_key)
        print(f"✅ Файл {local_path} успешно загружен в S3 как {s3_key}")
    except Exception as e:
        print(f"❌ Ошибка загрузки в S3: {e}")

# Если запускается как standalone
if __name__ == "__main__":
    # Путь к локальному CSV-файлу
    local_csv_path = Path("vlad/data/megafon_news.csv")

    if not local_csv_path.exists():
        print(f"❌ Файл не найден: {local_csv_path.resolve()}")
    else:
        # Ключ в бакете (имя в S3)
        s3_object_key = "data/features/megafon_news.csv"
        upload_to_s3(str(local_csv_path), s3_object_key)