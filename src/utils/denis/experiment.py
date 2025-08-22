import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import boto3
from io import BytesIO
import os
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from weather_data.main import create_new_weather_columns, expand_df_to_daily
from weather_data.config import regions_coords

load_dotenv()
RANDOM_STATE = 239
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
# MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET")

def main():
    # Инициализация MLflow эксперимента
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("NPS_Weather_RF_Regression")

    # s3_client = boto3.client(
    #     's3',
    #     aws_access_key_id=AWS_ACCESS_KEY_ID,
    #     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    #     endpoint_url=S3_ENDPOINT_URL
    # )

    with mlflow.start_run():
        # Загружаем данные (локально или из S3, раскомментируйте, если используете S3)
        # bucket_name = MLFLOW_S3_BUCKET
        # object_key = 'data/target/выгрузка для ЦУ.xlsx'
        #
        # response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        # data = response['Body'].read()
        # df = pd.read_excel(BytesIO(data), sheet_name="data cleaned")

        df = pd.read_excel("data/target.xlsx", sheet_name="data cleaned")

        df.drop("start_date", axis=1, inplace=True)
        df = create_new_weather_columns(df[:50000], cache_path='denis/cache/weather_cache.pkl')
        feature_cols = [col for col in df.columns if col not in ["year", "month", "nps", "state"]]
        X = df[feature_cols]
        y = df["nps"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        model = RandomForestRegressor(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Логируем параметры и метрики
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Создаем signature для логирования модели
        input_example = X_train.head(5)
        signature = infer_signature(X_train, model.predict(X_train))

        # Логируем модель с подписями и примером входных данных
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()
