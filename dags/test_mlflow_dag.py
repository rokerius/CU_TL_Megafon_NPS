from datetime import datetime, timedelta
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from src.utils.utils import init_mlflow, get_experiment_artifacts, get_experiment_metrics, log_experiment
import os
import mlflow

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_model():
    """Тестирование модели из MLflow"""
    # Инициализация MLflow
    init_mlflow(
        tracking_uri='http://mlflow:5001',
        experiment_name="test_experiment_utils",
        s3_bucket="megafon-nps"
    )
    
    # Получение последнего запуска из эксперимента
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("model_comparison")
    if not experiment:
        raise ValueError("Эксперимент 'model_comparison' не найден")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("Запуски не найдены в эксперименте")
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    
    # Получение артефактов последнего запуска
    artifacts = get_experiment_artifacts(run_id)
    if not artifacts:
        raise ValueError("Артефакты не найдены для последнего запуска")
    
    model = artifacts.get("model")
    if model is None:
        raise ValueError("Модель не найдена в артефактах")
    
    # Создание тестовых данных
    X_test = np.array([
        [6,7, 8, 9, 10], 
        [7,8, 9, 10, 11], 
        [8,9, 10, 11, 12], 
        [9,10, 11, 12, 13],
        [10,11, 12, 13, 14]
    ])
    
    # Выполнение предсказаний
    y_pred = model.predict(X_test)
    
    # Получение метрик оригинальной модели
    metrics = get_experiment_metrics(run_id)
    
    # Логирование результатов теста
    new_run_id = log_experiment(
        model=model,
        model_name="test_model",
        params={
            "model_type": "LinearRegression",
            "test_size": len(X_test)
        },
        metrics={
            "original_mse": metrics.get("mse", 0),
            "test_samples": len(X_test)
        },
        artifacts={
            "X_test": X_test,
            "y_pred": y_pred
        }
    )
    
    print(f"Тестирование завершено. Run ID: {new_run_id}")
    print(f"Предсказания: {y_pred}")
    return new_run_id

with DAG(
    'test_mlflow_model',
    default_args=default_args,
    description='Тестирование модели из MLflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Задача для тестирования модели
    test_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model,
    )
    
    # Задача для вывода результатов
    print_results = BashOperator(
        task_id='print_results',
        bash_command='echo "Тестирование модели завершено"',
    )
    
    # Определение порядка выполнения задач
    test_task >> print_results 