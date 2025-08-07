# Настройка MLflow в Airflow

## Что было сделано

1. **Создан кастомный образ Airflow с MLflow**
   - Создан `Dockerfile.airflow` для сборки кастомного образа
   - MLflow установлен непосредственно в образе
   - Обновлен `docker-compose.yaml` для использования кастомного образа

2. **Обновлены URI для подключения к MLflow**
   - В `dags/utils/utils.py` изменен URI по умолчанию с `http://localhost:5001` на `http://mlflow:5001`
   - В `dags/test_mlflow_dag.py` обновлен URI для корректной работы в Docker

3. **Создан тестовый DAG**
   - `dags/test_mlflow_simple.py` - простой тест подключения к MLflow

## Как запустить

1. **Сборка кастомного образа:**
   ```bash
   docker-compose build
   ```

2. **Запуск всех сервисов:**
   ```bash
   docker-compose up -d
   ```

2. **Проверка статуса:**
   ```bash
   docker-compose ps
   ```

3. **Доступ к веб-интерфейсам:**
   - Airflow: http://localhost:8080
   - MLflow: http://localhost:5001

## Использование MLflow в DAG

### Простой пример:

```python
import mlflow
from airflow.operators.python import PythonOperator

def train_model():
    # Установка URI для MLflow
    mlflow.set_tracking_uri("http://mlflow:5001")
    
    # Создание эксперимента
    mlflow.set_experiment("my_experiment")
    
    with mlflow.start_run():
        # Ваш код обучения модели
        mlflow.log_param("param1", "value1")
        mlflow.log_metric("accuracy", 0.95)
        mlflow.sklearn.log_model(model, "model")

# В DAG
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
)
```

### Использование утилит из utils.py:

```python
from utils.utils import init_mlflow, log_experiment

def my_task():
    # Инициализация MLflow
    init_mlflow(
        tracking_uri='http://mlflow:5001',
        experiment_name="my_experiment",
        s3_bucket="megafon-nps"
    )
    
    # Логирование эксперимента
    run_id = log_experiment(
        model=my_model,
        model_name="my_model",
        params={"param1": "value1"},
        metrics={"accuracy": 0.95}
    )
```

## Переменные окружения

Основные переменные для MLflow:
- `MLFLOW_S3_BUCKET=megafon-nps` - S3 бакет для артефактов
- `S3_ENDPOINT_URL=https://storage.yandexcloud.net` - S3 эндпоинт
- `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` - ключи доступа

## Проверка работы

1. Запустите тестовый DAG `test_mlflow_simple`
2. Проверьте логи выполнения
3. Откройте MLflow UI и убедитесь, что эксперимент создался

## Возможные проблемы

1. **Ошибка подключения к MLflow:**
   - Убедитесь, что контейнер mlflow запущен: `docker-compose ps mlflow`
   - Проверьте логи: `docker-compose logs mlflow`

2. **Ошибка сборки кастомного образа:**
   - Пересоберите образы: `docker-compose build --no-cache`
   - Проверьте, что все зависимости указаны в requirements.txt и Dockerfile.airflow

3. **Проблемы с правами доступа:**
   - Убедитесь, что AIRFLOW_UID установлен правильно
   - Проверьте права на директории: `ls -la dags/ logs/ config/ plugins/`

3. **Проблемы с S3:**
   - Проверьте правильность ключей доступа
   - Убедитесь, что бакет существует и доступен 