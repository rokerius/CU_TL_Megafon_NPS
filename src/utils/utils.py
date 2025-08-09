# utils.py
# Helper functions

import os
import mlflow
import mlflow.sklearn
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
from io import BytesIO, StringIO
import logging

logger = logging.getLogger(__name__)

def init_mlflow(
    tracking_uri: str = "http://mlflow:5001",
    experiment_name: str = "default_experiment",
    s3_endpoint_url: str = "https://storage.yandexcloud.net",
    s3_bucket: str = "megafon-nps"
) -> None:
    """
    Инициализация подключения к MLflow
    
    Args:
        tracking_uri: URI MLflow сервера
        experiment_name: Название эксперимента
        s3_endpoint_url: URL S3 хранилища
        s3_bucket: Название S3 бакета
    """
    # Установка URI для MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    # Настройка S3
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    os.environ['MLFLOW_S3_BUCKET'] = s3_bucket
    os.environ['MLFLOW_ARTIFACT_ROOT'] = f's3://{s3_bucket}/mlflow'
    
    # Создание эксперимента, если не существует
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def log_experiment(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Логирование эксперимента в MLflow
    
    Args:
        model: Обученная модель
        model_name: Название модели
        params: Параметры модели
        metrics: Метрики модели
        artifacts: Дополнительные артефакты
        run_name: Название запуска
        
    Returns:
        str: ID запуска
    """
    if run_name is None:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Логирование параметров
        mlflow.log_params(params)
        
        # Логирование метрик
        mlflow.log_metrics(metrics)
        
        # Логирование модели
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Логирование дополнительных артефактов
        if artifacts:
            for name, artifact in artifacts.items():
                if isinstance(artifact, (pd.DataFrame, np.ndarray)):
                    mlflow.log_dict(artifact.to_dict() if isinstance(artifact, pd.DataFrame) else artifact.tolist(), 
                                  f"{name}.json")
                else:
                    mlflow.log_dict(artifact, f"{name}.json")
        
        return run.info.run_id

def get_experiment_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Получение артефактов эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, Any]: Словарь с артефактами
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = {}
    
    # Получение информации о запуске
    run = client.get_run(run_id)
    
    # Получение артефактов
    for artifact in client.list_artifacts(run_id):
        if artifact.path.endswith('.json'):
            artifacts[artifact.path] = client.download_artifacts(run_id, artifact.path)
    
    # Получение модели
    model_path = f"runs:/{run_id}/model"
    artifacts['model'] = mlflow.sklearn.load_model(model_path)
    
    return artifacts

def get_experiment_metrics(run_id: str) -> Dict[str, float]:
    """
    Получение метрик эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics

def get_experiment_params(run_id: str) -> Dict[str, str]:
    """
    Получение параметров эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, str]: Словарь с параметрами
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params

def get_artifacts_by_run_name(run_name: str, experiment_name: str = None) -> Dict[str, Any]:
    """
    Получение артефактов по имени запуска
    
    Args:
        run_name: Имя запуска
        experiment_name: Имя эксперимента (опционально)
        
    Returns:
        Dict[str, Any]: Словарь с артефактами
    """
    client = mlflow.tracking.MlflowClient()
    
    # Если указано имя эксперимента, ищем в нем
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Эксперимент '{experiment_name}' не найден")
        experiment_id = experiment.experiment_id
    else:
        experiment_id = None
    
    # Поиск запуска по имени
    runs = client.search_runs(
        experiment_ids=[experiment_id] if experiment_id else None,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"Запуск с именем '{run_name}' не найден")
    
    run = runs[0]
    run_id = run.info.run_id
    
    # Получение артефактов
    artifacts = {}
    
    # Получение всех артефактов
    for artifact in client.list_artifacts(run_id):
        if artifact.path.endswith('.json'):
            artifacts[artifact.path] = client.download_artifacts(run_id, artifact.path)
    
    # Получение модели
    try:
        model_path = f"runs:/{run_id}/model"
        artifacts['model'] = mlflow.sklearn.load_model(model_path)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
    
    return artifacts

def get_file_format(key: str) -> str:
    """
    Determine file format from S3 key.
    
    Parameters
    ----------
    key : str
        S3 object key (path to file)
        
    Returns
    -------
    str
        File format ('csv', 'parquet', 'json', 'excel')
        
    Raises
    ------
    ValueError
        If file format is not supported
    """
    # Get file extension
    ext = os.path.splitext(key)[1].lower()
    
    # Map extension to format
    format_map = {
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.json': 'json',
        '.xlsx': 'excel',
        '.xls': 'excel'
    }
    
    if ext not in format_map:
        raise ValueError(f"Unsupported file format: {ext}")
        
    return format_map[ext]

def read_from_s3(
    s3_client: boto3.client,
    bucket: str,
    key: str,
    file_format: Optional[str] = None,
    **kwargs: Dict[str, Any]
) -> Union[pd.DataFrame, None]:
    """
    Read data from S3 bucket into pandas DataFrame.
    
    Parameters
    ----------
    s3_client : boto3.client
        Initialized S3 client
    bucket : str
        S3 bucket name
    key : str
        S3 object key (path to file)
    file_format : str, optional
        File format ('csv', 'parquet', 'json', 'excel'). If None, will be determined from key.
    **kwargs : Dict[str, Any]
        Additional arguments to pass to pandas read function
        
    Returns
    -------
    Union[pd.DataFrame, None]
        DataFrame with data or None if error occurred
    """
    try:
        # Determine file format if not provided
        if file_format is None:
            file_format = get_file_format(key)
        
        # Get object from S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Read data based on file format
        if file_format.lower() == 'csv':
            df = pd.read_csv(BytesIO(response['Body'].read()), **kwargs)
        elif file_format.lower() == 'parquet':
            df = pd.read_parquet(BytesIO(response['Body'].read()), **kwargs)
        elif file_format.lower() == 'json':
            df = pd.read_json(BytesIO(response['Body'].read()), **kwargs)
        elif file_format.lower() == 'excel':
            df = pd.read_excel(BytesIO(response['Body'].read()), **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        logger.info(f"Successfully read {file_format} file from s3://{bucket}/{key}")
        return df
        
    except Exception as e:
        logger.error(f"Error reading from S3: {str(e)}")
        return None

def write_to_s3(
    s3_client: boto3.client,
    df: pd.DataFrame,
    bucket: str,
    key: str,
    file_format: Optional[str] = None,
    **kwargs: Dict[str, Any]
) -> bool:
    """
    Write DataFrame to S3 bucket.
    
    Parameters
    ----------
    s3_client : boto3.client
        Initialized S3 client
    df : pd.DataFrame
        DataFrame to write
    bucket : str
        S3 bucket name
    key : str
        S3 object key (path to file)
    file_format : str, optional
        File format ('csv', 'parquet', 'json', 'excel'). If None, will be determined from key.
    **kwargs : Dict[str, Any]
        Additional arguments to pass to pandas write function
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Determine file format if not provided
        if file_format is None:
            file_format = get_file_format(key)
        
        # Convert DataFrame to bytes based on file format
        buffer = BytesIO()
        if file_format.lower() == 'csv':
            df.to_csv(buffer, **kwargs)
        elif file_format.lower() == 'parquet':
            df.to_parquet(buffer, **kwargs)
        elif file_format.lower() == 'json':
            df.to_json(buffer, **kwargs)
        elif file_format.lower() == 'excel':
            df.to_excel(buffer, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
            
        # Upload to S3
        buffer.seek(0)
        s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        
        logger.info(f"Successfully wrote {file_format} file to s3://{bucket}/{key}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to S3: {str(e)}")
        return False 