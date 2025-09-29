# upload_excel_to_s3_dag.py
from __future__ import annotations
import logging
import pendulum
import boto3
import os
from airflow.decorators import dag, task
from airflow.models import Variable

BUCKET = Variable.get("s3_bucket", "cu-mf-project")
LOCAL_PATH = Variable.get("local_xlsx_path", "/opt/airflow/data/target.xlsx")
S3_KEY = Variable.get("s3_xlsx_path", "raw/target.xlsx")


@dag(
    dag_id="D_upload_excel_to_s3",
    description="Загружает локальный Excel-файл в S3 (Yandex Object Storage)",
    catchup=False,
    tags=["upload", "s3", "excel"],
)
def D_upload_excel_to_s3_dag():

    @task
    def upload_file():
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )


        logging.info("Uploading %s to s3://%s/%s", LOCAL_PATH, BUCKET, S3_KEY)
        s3.upload_file(LOCAL_PATH, BUCKET, S3_KEY)
        logging.info("✅ File uploaded successfully")

    upload_file()


D_upload_excel_to_s3_dag()
