# Исправление проблемы с S3 credentials в Airflow

## Проблема
DAG `target_dag.py` падал с ошибкой `NoCredentialsError: Unable to locate credentials` при попытке подключения к S3.

## Причина
Код пытался использовать AWS credentials, но на самом деле используется Yandex Cloud Object Storage, который требует других настроек.

## Исправления

### 1. Обновлен docker-compose.yaml
Добавлены правильные переменные окружения для Yandex Cloud:
```yaml
# Yandex Cloud Object Storage credentials
AWS_ACCESS_KEY_ID: ${S3_ACCESS_KEY}
AWS_SECRET_ACCESS_KEY: ${S3_SECRET_KEY}
AWS_DEFAULT_REGION: ru-central1
# Yandex Cloud S3 endpoint
S3_ENDPOINT_URL: ${S3_ENDPOINT}
```

### 2. Обновлен DAG target_dag.py
- Добавлена поддержка Yandex Cloud Object Storage
- Используется boto3 клиент с правильным endpoint
- Добавлена обработка ошибок и логирование

### 3. Создан альтернативный DAG target_dag_fixed.py
Содержит исправленную версию с правильной конфигурацией для Yandex Cloud.

### 4. Создан скрипт run_airflow.sh
Автоматически загружает переменные окружения из `megafon-access.env` и запускает Airflow.

## Как использовать

### Вариант 1: Использовать исправленный оригинальный DAG
1. Запустите Airflow с правильными переменными окружения:
```bash
./run_airflow.sh
```

2. Или вручную загрузите переменные и запустите:
```bash
export $(cat megafon-access.env | grep -v '^#' | xargs)
docker-compose up -d
```

### Вариант 2: Использовать новый DAG
Используйте `target_dag_fixed.py`, который имеет правильную конфигурацию изначально.

## Проверка
После запуска проверьте:
1. Airflow UI доступен на http://localhost:8080
2. DAG `download_s3_target_files` или `download_s3_target_files_fixed` виден в списке
3. При запуске DAG успешно подключается к Yandex Cloud и скачивает файлы

## Переменные окружения
Убедитесь, что в `megafon-access.env` есть:
- `S3_ENDPOINT` - endpoint Yandex Cloud Object Storage
- `S3_ACCESS_KEY` - ключ доступа
- `S3_SECRET_KEY` - секретный ключ
- `S3_BUCKET` - имя bucket'а 