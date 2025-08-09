# CU_TL_Megafon_NPS - Проект для работы с NPS данными

Этот проект представляет собой комплексное решение для обработки и анализа NPS (Net Promoter Score) данных с использованием Apache Airflow для оркестрации задач и MLflow для управления машинным обучением.

## 🏗️ Архитектура проекта

Проект состоит из следующих компонентов:
- **Apache Airflow** - оркестрация и планирование задач
- **MLflow** - управление экспериментами машинного обучения
- **PostgreSQL** - база данных для Airflow
- **Redis** - брокер сообщений для Celery
- **S3-совместимое хранилище** - для хранения данных и артефактов

## 📋 Предварительные требования

### Системные требования
- **Операционная система**: macOS, Linux или Windows (с WSL2)
- **Docker**: версия 20.10+ 
- **Docker Compose**: версия 2.0+
- **Python**: 3.10 или 3.11 (для локальной разработки)
- **Git**: для клонирования репозитория

### Минимальные ресурсы
- **RAM**: минимум 4GB (рекомендуется 8GB+)
- **CPU**: минимум 2 ядра
- **Дисковое пространство**: минимум 10GB свободного места

## 🚀 Пошаговая инструкция запуска

### Шаг 1: Клонирование репозитория

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd CU_TL_Megafon_NPS
```

### Шаг 2: Настройка переменных окружения

Создайте файл `.env` в корневой директории проекта:

```bash
# Создание файла .env
touch .env
```

Добавьте в файл `.env` следующие переменные:

```env
# Airflow настройки
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow

# AWS/S3 настройки
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=your_region
S3_ENDPOINT_URL=your_s3_endpoint_url
MLFLOW_S3_BUCKET=your_s3_bucket_name

# Дополнительные настройки (опционально)
_PIP_ADDITIONAL_REQUIREMENTS=""
```

**⚠️ Важно**: Замените значения на ваши реальные данные доступа к S3-совместимому хранилищу.

### Шаг 3: Настройка прав доступа (только для Linux/macOS)

```bash
# Установка правильных прав доступа для Airflow
mkdir -p ./logs ./plugins ./config
chmod -R 777 ./logs ./plugins ./config
```

### Шаг 4: Запуск проекта с помощью Docker Compose

```bash
# Инициализация и запуск всех сервисов
docker-compose up -d

# Проверка статуса сервисов
docker-compose ps
```

### Шаг 5: Ожидание инициализации

После запуска подождите 2-3 минуты для полной инициализации всех сервисов. Вы можете отслеживать логи:

```bash
# Просмотр логов всех сервисов
docker-compose logs -f

# Просмотр логов конкретного сервиса
docker-compose logs -f airflow-init
```

### Шаг 6: Проверка доступности сервисов

После успешной инициализации вы сможете получить доступ к:

- **Airflow Web UI**: http://localhost:8080
  - Логин: `airflow`
  - Пароль: `airflow`

- **MLflow UI**: http://localhost:5001

## 🔧 Дополнительные команды

### Управление сервисами

```bash
# Остановка всех сервисов
docker-compose down

# Перезапуск сервисов
docker-compose restart

# Просмотр логов
docker-compose logs -f [service_name]

# Выполнение команд в контейнере
docker-compose exec airflow-worker airflow dags list
```

### Локальная разработка (опционально)

Если вы хотите работать с проектом локально без Docker:

```bash
# Запуск скрипта настройки окружения
chmod +x setup_env.sh
./setup_env.sh

# Активация виртуального окружения
source .venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

## 📁 Структура проекта

```
CU_TL_Megafon_NPS/
├── config/                 # Конфигурационные файлы Airflow
├── dags/                   # DAG файлы для Airflow
│   ├── target_dag.py      # Основной DAG для обработки данных
│   ├── test_mlflow_dag.py # Тестовый DAG для MLflow
│   └── ...
├── mlflow/                 # Конфигурация MLflow
│   ├── Dockerfile         # Docker образ для MLflow
│   ├── requirements.txt   # Зависимости MLflow
│   └── mlruns/           # Локальные эксперименты MLflow
├── logs/                  # Логи Airflow
├── plugins/               # Плагины Airflow
├── src/                   # Исходный код проекта
├── tests/                 # Тесты
├── docker-compose.yaml    # Конфигурация Docker Compose
├── Dockerfile.airflow     # Docker образ для Airflow
├── requirements.txt       # Python зависимости
├── setup_env.sh          # Скрипт настройки окружения
└── README.md             # Этот файл
```

## 🔍 Мониторинг и отладка

### Проверка здоровья сервисов

```bash
# Проверка статуса всех контейнеров
docker-compose ps

# Проверка здоровья Airflow
curl http://localhost:8080/health

# Проверка здоровья MLflow
curl http://localhost:5001/health
```

### Просмотр логов

```bash
# Логи инициализации
docker-compose logs airflow-init

# Логи планировщика
docker-compose logs airflow-scheduler

# Логи веб-сервера
docker-compose logs airflow-apiserver

# Логи MLflow
docker-compose logs mlflow
```

## 🐛 Устранение неполадок

### Частые проблемы

1. **Ошибка "Permission denied"**
   ```bash
   # Решение для Linux/macOS
   sudo chown -R $USER:$USER ./logs ./plugins ./config
   ```

2. **Сервисы не запускаются**
   ```bash
   # Проверка ресурсов Docker
   docker system df
   
   # Очистка неиспользуемых ресурсов
   docker system prune -a
   ```

3. **Проблемы с подключением к S3**
   - Проверьте правильность переменных окружения в `.env`
   - Убедитесь, что S3-совместимое хранилище доступно

4. **Airflow не инициализируется**
   ```bash
   # Принудительная переинициализация
   docker-compose down -v
   docker-compose up -d
   ```

### Полезные команды для отладки

```bash
# Проверка переменных окружения в контейнере
docker-compose exec airflow-worker env | grep AIRFLOW

# Проверка подключения к базе данных
docker-compose exec airflow-worker airflow db check

# Список DAG'ов
docker-compose exec airflow-worker airflow dags list

# Проверка подключений
docker-compose exec airflow-worker airflow connections list
```

## 📚 Дополнительная документация

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте раздел "Устранение неполадок"
2. Изучите логи сервисов
3. Убедитесь, что все переменные окружения настроены правильно
4. Проверьте системные требования

## 📝 Лицензия

[Укажите лицензию проекта]

---

**Примечание**: Этот README содержит базовые инструкции по запуску проекта. Для продакшн-развертывания рекомендуется дополнительная настройка безопасности и мониторинга.
