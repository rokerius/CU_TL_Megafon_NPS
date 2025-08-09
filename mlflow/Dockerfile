FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/mlruns
RUN useradd -ms /bin/bash mlflowuser
USER mlflowuser

EXPOSE 5001
ENV MLFLOW_BACKEND_STORE_URI=/app/mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://megafon-nps/mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5001", "--serve-artifacts"]
