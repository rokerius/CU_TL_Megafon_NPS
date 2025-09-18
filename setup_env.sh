#!/bin/bash

set -e

# 1. Проверка наличия Python 3.10+ (или 3.11)
PYTHON_BIN="python3"
PYTHON_VERSION_REQUIRED="3.10"

if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
    PYTHON_VERSION_REQUIRED="3.11"
elif command -v python3.10 &>/dev/null; then
    PYTHON_BIN="python3.10"
    PYTHON_VERSION_REQUIRED="3.10"
fi

PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')

if [[ $(echo -e "$PYTHON_VERSION\n$PYTHON_VERSION_REQUIRED" | sort -V | head -n1) != "$PYTHON_VERSION_REQUIRED" ]]; then
    echo "[ERROR] Требуется Python $PYTHON_VERSION_REQUIRED или выше. Найдено: $PYTHON_VERSION."
    exit 1
fi

echo "[INFO] Используется $PYTHON_BIN ($PYTHON_VERSION)"

# 2. Создание виртуального окружения
if [ ! -d ".venv" ]; then
    echo "[INFO] Создаю виртуальное окружение .venv"
    $PYTHON_BIN -m venv .venv
else
    echo "[INFO] Виртуальное окружение .venv уже существует"
fi

# 3. Активация виртуального окружения
source .venv/bin/activate

echo "[INFO] Виртуальное окружение активировано"

# 4. Обновление pip
pip install --upgrade pip

# 5. Установка зависимостей
if [ -f "requirements.txt" ]; then
    echo "[INFO] Устанавливаю зависимости из requirements.txt"
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt не найден!"
fi

# 6. Инструкция по использованию .env файла
if [ -f "megafon-access.env" ]; then
    echo "[INFO] Найден файл megafon-access.env."
    echo "[INFO] Для экспорта переменных окружения выполните:"
    echo "    export \
$(grep -v '^#' megafon-access.env | xargs)"
    echo "или используйте dotenv/direnv для автоматической подгрузки .env файлов."
else
    echo "[WARNING] megafon-access.env не найден! Проверьте переменные окружения вручную."
fi

echo "[INFO] Установка завершена. Для активации окружения используйте:"
echo "    source .venv/bin/activate"
echo "[INFO] Для запуска проекта используйте инструкции из документации или docker-compose, если требуется."
