# feature_engineering_sqlite_dag.py
from __future__ import annotations
import logging
import pendulum
import pandas as pd
import sqlite3
import s3fs
from airflow.decorators import dag, task
from src.utils.denis.main import prepare_data


S3_BUCKET = "cu-mf-project"
RAW_PATH = f"s3://{S3_BUCKET}/raw/target.xlsx"
FEATURES_PATH = f"s3://{S3_BUCKET}/features/df_with_features_1.csv"


@dag(
    dag_id="D_feature_engineering_s3",
    description="Простейший расчёт признаков с хранением в S3 и SQLite (песочница)",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,   # запускать вручную
    catchup=False,
    tags=["ml", "features", "s3"],
)
def D_feature_engineering_s3_dag():

    @task
    def extract() -> pd.DataFrame:
        """Читаем Excel из S3"""
        logging.info("Reading raw Excel from %s", RAW_PATH)
        df = pd.read_excel(
            RAW_PATH, "data cleaned",
            storage_options={"client_kwargs": {"endpoint_url": "https://storage.yandexcloud.net"}},
        )
        logging.info("Raw dataset loaded: shape=%s", df.shape)
        return df

    @task
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляем новые признаки"""
        df_features = prepare_data(df, weather_cache_path='cache/weather_cache.pkl')
        logging.info("Features calculated: shape=%s", df_features.shape)
        return df_features

    @task
    def load_and_inspect(df: pd.DataFrame):
        """Сохраняем фичи в S3"""
        # 2. S3
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://storage.yandexcloud.net"})
        with fs.open(FEATURES_PATH, "w") as f:
            df.to_csv(f, index=False)
        logging.info("Uploaded features to %s", FEATURES_PATH)

        # 3. Проверим доступность файла
        with fs.open(FEATURES_PATH, "r") as f:
            head = "".join([next(f) for _ in range(5)])
        logging.info("=== Feature Store snapshot from S3 (%s) ===", FEATURES_PATH)
        logging.info("\n%s", head)
        logging.info("===========================================")

    raw_df = extract()
    feats_df = transform(raw_df)
    load_and_inspect(feats_df)


D_feature_engineering_s3_dag()
