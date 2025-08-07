from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import os
import boto3
from botocore.config import Config

S3_BUCKET = 'megafon-nps'
S3_PREFIX = 'data/target/'
LOCAL_DIR = '/opt/airflow/data/target'

def download_s3_files():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # Get environment variables for Yandex Cloud
    endpoint_url = os.getenv('S3_ENDPOINT_URL', 'https://storage.yandexcloud.net')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
    
    print(f"Connecting to Yandex Cloud Object Storage at {endpoint_url}")
    print(f"Bucket: {S3_BUCKET}, Prefix: {S3_PREFIX}")
    
    # Create S3 client for Yandex Cloud
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='ru-central1',
        config=Config(signature_version='s3v4')
    )
    
    try:
        # List objects in the bucket with the specified prefix
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX
        )
        
        if 'Contents' not in response:
            print(f"No files found in bucket {S3_BUCKET} with prefix {S3_PREFIX}")
            return
        
        files = response['Contents']
        print(f"Found {len(files)} files to download")
        
        for obj in files:
            key = obj['Key']
            filename = os.path.basename(key)
            local_path = os.path.join(LOCAL_DIR, filename)
            
            print(f"Downloading {key} to {local_path}")
            s3_client.download_file(S3_BUCKET, key, local_path)
            print(f"Successfully downloaded {filename}")
            
    except Exception as e:
        print(f"Error downloading files from Yandex Cloud: {e}")
        raise

with DAG(
    dag_id='download_s3_target_files',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['s3', 'download'],
) as dag:

    download = PythonOperator(
        task_id='download_s3_files',
        python_callable=download_s3_files,
    )