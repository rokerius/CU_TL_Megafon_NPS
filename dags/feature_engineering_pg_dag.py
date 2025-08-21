# feature_engineering_sqlite_dag.py
from __future__ import annotations
import logging
import pendulum
import pandas as pd
import sqlite3
from airflow.decorators import dag, task
from utils.main import create_new_weather_columns
from utils.denis.main import prepare_data

DB_PATH = "/tmp/ml_features.db"


# DAG для простейшего расчёта признаков с хранением в SQLite
@dag(
    dag_id="D_feature_engineering_sqlite",
    description="Простейший расчёт признаков с хранением в SQLite (песочница)",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    schedule=None,   # запускать вручную
    catchup=False,
    tags=["ml", "features", "sqlite"],
)
def D_feature_engineering_sqlite_dag():

    @task
    def extract() -> str:
        """Имитируем получение сырых данных: создаём dataframe и сохраняем в csv"""
        df = pd.DataFrame(
            {
                "start_date": ["Алтайский край"] * 5,
                "year": [2024] * 5,
                "month": [10] * 5,
                "nps": [8, 10, 10, 10, 10],
                "state": ["Алтайский край"] * 5,
            }
        )
        path = "/tmp/raw_events.csv"
        df.to_csv(path, index=False)
        logging.info("Synthetic raw data saved to %s", path)
        return path

    @task
    def transform(path: str) -> str:
        """Добавляем новый признак"""
        df = pd.read_csv(path)
        df = prepare_data(df)

        out_path = "/tmp/megafon_nps_features.csv"
        df.to_csv(out_path, index=False)
        logging.info("Features calculated: %s", df.shape)
        return out_path

    @task
    def load_and_inspect(path: str):
        df = pd.read_csv(path)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("megafon_nps", conn, if_exists="replace", index=False)

        logging.info("Inserted %s rows into SQLite megafon_nps", len(df))

        # сразу читаем и логируем
        df_read = pd.read_sql("SELECT * FROM megafon_nps", conn)
        logging.info("=== Feature Store snapshot (megafon_nps) ===")
        logging.info("\n%s", df_read.head().to_string(index=False))
        logging.info("===========================================")
        conn.close()
        
    raw = extract()
    feats = transform(raw)
    load_and_inspect(feats)




D_feature_engineering_sqlite_dag()
