# Устранение ошибок

## Ошибки сборки образа

### 1. Ошибка "COPY failed: file not found"
```
ERROR: failed to compute cache key: "/src" not found: not found
```

**Решение:** Убедитесь, что все файлы, указанные в Dockerfile.airflow, существуют:
- `requirements.txt`
- `setup.py`

### 2. Ошибка установки зависимостей
```
ERROR: Could not find a version that satisfies the requirement
```

**Решение:** 
1. Проверьте версии в requirements.txt
2. Обновите pip: `pip install --upgrade pip`
3. Пересоберите образ: `docker-compose build --no-cache`

### 3. Ошибка прав доступа
```
ERROR: permission denied
```

**Решение:**
1. Установите правильный AIRFLOW_UID:
   ```bash
   export AIRFLOW_UID=$(id -u)
   ```
2. Исправьте права на директории:
   ```bash
   sudo chown -R $AIRFLOW_UID:$AIRFLOW_UID dags/ logs/ config/ plugins/
   ```

## Ошибки запуска сервисов

### 1. Ошибка подключения к базе данных
```
ERROR: connection to server at "postgres" failed
```

**Решение:**
1. Убедитесь, что PostgreSQL запущен: `docker-compose ps postgres`
2. Проверьте логи: `docker-compose logs postgres`
3. Дождитесь готовности базы: `docker-compose logs airflow-init`

### 2. Ошибка подключения к Redis
```
ERROR: connection to Redis failed
```

**Решение:**
1. Убедитесь, что Redis запущен: `docker-compose ps redis`
2. Проверьте логи: `docker-compose logs redis`

### 3. Ошибка MLflow
```
ERROR: connection to MLflow failed
```

**Решение:**
1. Проверьте, что MLflow запущен: `docker-compose ps mlflow`
2. Проверьте логи: `docker-compose logs mlflow`
3. Убедитесь, что URI правильный: `http://mlflow:5001`

## Полезные команды

### Проверка статуса сервисов
```bash
docker-compose ps
```

### Просмотр логов
```bash
# Все сервисы
docker-compose logs

# Конкретный сервис
docker-compose logs airflow-scheduler
docker-compose logs mlflow
```

### Перезапуск сервисов
```bash
# Все сервисы
docker-compose restart

# Конкретный сервис
docker-compose restart airflow-scheduler
```

### Очистка и пересборка
```bash
# Остановить все сервисы
docker-compose down

# Удалить все образы и контейнеры
docker-compose down --rmi all --volumes --remove-orphans

# Пересобрать образы
docker-compose build --no-cache

# Запустить заново
docker-compose up -d
```

### Проверка версий
```bash
# Проверить версию Airflow в контейнере
docker-compose exec airflow-scheduler python -c "import airflow; print(airflow.__version__)"

# Проверить версию MLflow в контейнере
docker-compose exec airflow-scheduler python -c "import mlflow; print(mlflow.__version__)"
``` 