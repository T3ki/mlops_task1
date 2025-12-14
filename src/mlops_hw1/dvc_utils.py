
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "data" / "train_datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_dvc_env() -> dict:
    """Готовим env для DVC, чтобы были AWS_* переменные для Minio."""
    env = os.environ.copy()
    # Пробрасываем ключи из S3_* в AWS_*, если они ещё не заданы
    if "AWS_ACCESS_KEY_ID" not in env and "S3_ACCESS_KEY" in env:
        env["AWS_ACCESS_KEY_ID"] = env["S3_ACCESS_KEY"]
    if "AWS_SECRET_ACCESS_KEY" not in env and "S3_SECRET_KEY" in env:
        env["AWS_SECRET_ACCESS_KEY"] = env["S3_SECRET_KEY"]
    return env


def save_dataset_with_dvc(
    features: list[list[float]],
    target: list[int],
    model_id: str,
) -> Path:
    """
    Сохраняет датасет обучения в CSV, добавляет его в DVC и пушит на Minio.

    :param features: признаки
    :param target: целевая переменная
    :param model_id: ID модели (чтобы связать датасет и модель)
    :return: путь к локальному CSV-файлу
    """
    # 1. Сохраняем данные в CSV
    df = pd.DataFrame(features)
    df["target"] = target

    dataset_path = DATASETS_DIR / f"train_{model_id}.csv"
    df.to_csv(dataset_path, index=False)
    logger.info("Saved training dataset locally: %s", dataset_path)

    # 2. DVC add + push
    rel_path = dataset_path.relative_to(PROJECT_ROOT)
    env = _prepare_dvc_env()

    try:

        subprocess.run(
            [sys.executable, "-m", "dvc", "add", str(rel_path)],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
        subprocess.run(
            [sys.executable, "-m", "dvc", "push"],
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
        logger.info("Dataset tracked by DVC and pushed to remote: %s", rel_path)
    except subprocess.CalledProcessError as e:
        logger.error("DVC command failed for %s: %s", rel_path, e)

    return dataset_path