from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_mlflow_connection():
    """Простой тест подключения к MLflow"""
    try:
        # Установка URI для MLflow
        mlflow.set_tracking_uri("http://mlflow:5001")
        
        # Создание эксперимента
        experiment_name = "airflow_test"
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        
        # Создание простых данных
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([2, 4, 6, 8])
        
        # Обучение простой модели
        model = LinearRegression()
        model.fit(X, y)
        
        # Предсказания
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        # Логирование в MLflow
        with mlflow.start_run(run_name="airflow_test_run") as run:
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("n_samples", len(X))
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", model.score(X, y))
            
            # Логирование модели
            mlflow.sklearn.log_model(model, "model")
            
            print(f"MLflow подключение успешно! Run ID: {run.info.run_id}")
            print(f"MSE: {mse}")
            print(f"R2 Score: {model.score(X, y)}")
            
        return run.info.run_id
        
    except Exception as e:
        print(f"Ошибка при подключении к MLflow: {e}")
        raise e

with DAG(
    'test_mlflow_simple',
    default_args=default_args,
    description='Простой тест MLflow в Airflow',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    test_mlflow = PythonOperator(
        task_id='test_mlflow_connection',
        python_callable=test_mlflow_connection,
    ) 