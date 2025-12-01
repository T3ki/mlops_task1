# src/mlops_hw1/storage.py
import logging
import os
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

logger = logging.getLogger(__name__)

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin123")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS", "mlops-hw1-models")

_s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def upload_model_to_s3(local_path: Path, model_id: str) -> None:
    """Загрузить модель в S3 (Minio)."""
    key = f"models/{model_id}.joblib"
    _s3_client.upload_file(str(local_path), S3_BUCKET_MODELS, key)
    logger.info(
        "Uploaded model %s to S3: bucket=%s, key=%s",
        model_id,
        S3_BUCKET_MODELS,
        key,
    )


def download_model_from_s3(local_path: Path, model_id: str) -> None:
    """Скачать модель из S3 (Minio) в локальный файл."""
    key = f"models/{model_id}.joblib"
    _s3_client.download_file(S3_BUCKET_MODELS, key, str(local_path))
    logger.info(
        "Downloaded model %s from S3 to %s",
        model_id,
        local_path,
    )


def delete_model_from_s3(model_id: str) -> None:
    """Удалить модель из S3 (Minio)."""
    key = f"models/{model_id}.joblib"
    _s3_client.delete_object(Bucket=S3_BUCKET_MODELS, Key=key)
    logger.info(
        "Deleted model %s from S3: bucket=%s, key=%s",
        model_id,
        S3_BUCKET_MODELS,
        key,
    )