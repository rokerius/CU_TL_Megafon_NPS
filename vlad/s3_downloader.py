#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from datetime import datetime

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv


# ---------------------------
# 1)  Конфиг и креды из .env
# ---------------------------
# ожидаемые переменные в config/megafon-access.env:
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_DEFAULT_REGION=ru-central1
# MLFLOW_S3_BUCKET=cu-mf-project
# MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
# (необяз.) S3_DEFAULT_PREFIX=   # например 'features/' или 'mlflow/'
ENV_PATH = "config/megafon-access.env"
load_dotenv(ENV_PATH)

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION    = os.getenv("AWS_DEFAULT_REGION", "ru-central1")
S3_ENDPOINT_URL       = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
BUCKET                = os.getenv("MLFLOW_S3_BUCKET")
DEFAULT_PREFIX        = os.getenv("S3_DEFAULT_PREFIX", "")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not BUCKET:
    print(f"[ERROR] Проверь {ENV_PATH}: нужны AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MLFLOW_S3_BUCKET", file=sys.stderr)
    sys.exit(1)

# Клиент S3
boto_cfg = Config(
    region_name=AWS_DEFAULT_REGION,
    signature_version="s3v4",
    s3={"addressing_style": "path"},
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=10,
    read_timeout=60,
)
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=boto_cfg,
)

transfer_cfg = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=8,
    use_threads=True,
)


# ---------------------------
# 2)  Утилиты
# ---------------------------
def ensure_dir(local_path: str) -> None:
    d = os.path.dirname(local_path)
    if d:
        os.makedirs(d, exist_ok=True)

def stat_line(key: str) -> str:
    try:
        head = s3.head_object(Bucket=BUCKET, Key=key)
        size = head.get("ContentLength", 0)
        lm   = head.get("LastModified")
        size_mb = f"{size/1024/1024:.2f} MB"
        lm_str = lm.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if isinstance(lm, datetime) else "?"
        return f"{key} | {size_mb} | {lm_str}"
    except ClientError as e:
        return f"{key} | <нет доступа/не найдено: {e.response.get('Error', {}).get('Code','?')}>"

def build_local_path(out_dir: str, key: str, flatten: bool) -> str:
    return os.path.join(out_dir, os.path.basename(key) if flatten else key.replace("\\", "/"))

def read_manifest(path: str) -> list[str]:
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            keys.append(line)
    return keys


# ---------------------------
# 3)  CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Скачать выбранные объекты из S3 в локальную папку.")
    p.add_argument("--out", default="data", help="Локальная папка назначения (по умолчанию: ./data)")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help=f"Префикс, который будет добавлен ко всем ключам (по умолчанию из .env: '{DEFAULT_PREFIX}')")
    p.add_argument("--keys", nargs="*", default=[],
                   help="Список ключей в бакете (через пробел). Пример: features/train.csv raw/target.xlsx")
    p.add_argument("--manifest", default=None,
                   help="Путь к файлу-списку ключей (по одному на строку).")
    p.add_argument("--flatten", action="store_true",
                   help="Сохранять только имена файлов (без подпапок S3).")
    p.add_argument("--dry-run", action="store_true",
                   help="Только показать, что будет скачано, без скачивания.")
    return p.parse_args()


# ---------------------------
# 4)  Основная логика
# ---------------------------
def main():
    args = parse_args()

    # собрать список ключей
    keys = list(args.keys)
    if args.manifest:
        keys += read_manifest(args.manifest)

    if not keys:
        print("[ERROR] Не переданы ключи (--keys) и не указан --manifest", file=sys.stderr)
        sys.exit(2)

    # применить префикс, если задан
    if args.prefix:
        keys = [ (args.prefix.rstrip("/") + "/" + k.lstrip("/")) for k in keys ]

    print(f"[INFO] Bucket:   {BUCKET}")
    print(f"[INFO] Endpoint: {S3_ENDPOINT_URL}")
    print(f"[INFO] Region:   {AWS_DEFAULT_REGION}")
    print(f"[INFO] Out dir:  {os.path.abspath(args.out)}")
    print(f"[INFO] Flatten:  {args.flatten}")
    print(f"[INFO] Dry-run:  {args.dry_run}")
    print("[INFO] Объекты:")
    for k in keys:
        print(" -", stat_line(k))

    if args.dry_run:
        print("[INFO] DRY-RUN: загрузка не выполняется.")
        return

    # качаем
    for k in keys:
        local_path = build_local_path(args.out, k if not args.prefix else k[len(args.prefix.rstrip('/') + '/') :], args.flatten)
        ensure_dir(local_path)
        print(f"[DL] {k}  ->  {local_path}")
        try:
            s3.download_file(BUCKET, k, local_path, Config=transfer_cfg)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "?")
            print(f"[ERR] Не удалось скачать {k}: {code} — {e}", file=sys.stderr)

    print("[OK] Готово.")


if __name__ == "__main__":
    main()