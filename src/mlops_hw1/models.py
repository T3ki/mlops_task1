import joblib
import uuid
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any
import pandas as pd
from .dvc_utils import save_dataset_with_dvc

from .storage import (
    upload_model_to_s3,
    download_model_from_s3,
    delete_model_from_s3,
)

from .tracking import init_mlflow, mlflow, MLFLOW_AVAILABLE
from sklearn.metrics import accuracy_score

# Создаем папку для хранения моделей, если ее нет
MODELS_DIR = Path("models_storage")
MODELS_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
}

def get_available_models() -> List[str]:
    """Возвращает список имен доступных моделей."""
    return list(AVAILABLE_MODELS.keys())

def train_model(
    model_name: str,
    hyperparams: Dict[str, Any],
    features: List[List[float]],
    target: List[int]
) -> str:
    """
    Обучает модель и сохраняет ее, плюс трекает обучение в MLflow.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not available.")

    # Генерируем ID модели, чтобы использовать его и для датасета, и для файла модели
    model_id = f"{model_name}_{uuid.uuid4().hex[:8]}"

    # Сохраняем датасет и отправляем его в DVC/Minio
    save_dataset_with_dvc(features, target, model_id)

    model_class = AVAILABLE_MODELS[model_name]
    model = model_class(**hyperparams)

    # Попробуем трекать обучение в MLflow
    if MLFLOW_AVAILABLE and init_mlflow():
        try:
            with mlflow.start_run(run_name=model_id):
                # Логируем параметры
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(hyperparams)
                mlflow.log_param("model_id", model_id)
                mlflow.log_metric("train_samples", len(target))
                if features:
                    mlflow.log_metric("train_features_dim", len(features[0]))

                # Обучаем модель
                model.fit(features, target)

                # Метрика на train
                y_pred = model.predict(features)
                acc = accuracy_score(target, y_pred)
                mlflow.log_metric("train_accuracy", acc)

                # Сохраняем модель локально
                save_path = MODELS_DIR / f"{model_id}.joblib"
                joblib.dump(model, save_path)
                upload_model_to_s3(save_path, model_id)

                # Логируем модель в MLflow
                mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.error("MLflow tracking failed, training without MLflow: %s", e)
            # обучаем без MLflow
            model.fit(features, target)
            save_path = MODELS_DIR / f"{model_id}.joblib"
            joblib.dump(model, save_path)
            upload_model_to_s3(save_path, model_id)
    else:
        # MLflow недоступен — просто обучаем модель
        model.fit(features, target)
        save_path = MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model, save_path)
        upload_model_to_s3(save_path, model_id)

    return model_id


def predict_with_model(model_id: str, features: List[List[float]]) -> List[int]:
    """
    Делает предсказание с помощью обученной модели.

    :param model_id: ID модели для использования.
    :param features: Признаки для предсказания.
    :return: Список с предсказаниями.
    """
    save_path = MODELS_DIR / f"{model_id}.joblib"

    if not save_path.exists():
        # Пробуем скачать модель из S3
        try:
            download_model_from_s3(save_path, model_id)
        except Exception as e:
            raise FileNotFoundError(
                f"Model with id '{model_id}' not found locally or in S3. {e}"
            )

    # Загружаем обученную модель
    model = joblib.load(save_path)

    # Делаем предсказание
    predictions = model.predict(features)

    # .tolist() конвертирует numpy array в обычный список,
    # что лучше для JSON-ответов
    return predictions.tolist()

def delete_model(model_id: str) -> None:
    """
    Удаляет сохраненную модель.

    :param model_id: ID модели для удаления.
    """
    model_path = MODELS_DIR / f"{model_id}.joblib"

    # Удаляем локальный файл, если есть
    if model_path.exists():
        os.remove(model_path)

    # Удаляем из S3
    delete_model_from_s3(model_id)

def retrain_model(
    model_id: str,
    hyperparams: Dict[str, Any],
    features: List[List[float]],
    target: List[int]
) -> None:
    """
    Переобучает существующую модель на новых данных и/или с новыми гиперпараметрами.
    Логирует переобучение в MLflow.
    """
    save_path = MODELS_DIR / f"{model_id}.joblib"
    if not save_path.exists():
        raise FileNotFoundError(f"Model with id '{model_id}' not found for retraining.")

    # Извлекаем имя класса модели из ID, например, из "logreg_a1b2c3d4" -> "logreg"
    try:
        model_name = model_id.split('_')[0]
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type '{model_name}' in model_id.")
    except IndexError:
        raise ValueError(f"Invalid model_id format: '{model_id}'. Expected 'name_uuid'.")

    model_class = AVAILABLE_MODELS[model_name]
    model = model_class(**hyperparams)

    # Попробуем трекать переобучение в MLflow
    if MLFLOW_AVAILABLE and init_mlflow():
        try:
            with mlflow.start_run(run_name=f"{model_id}_retrain"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(hyperparams)
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("run_type", "retrain")

                mlflow.log_metric("train_samples", len(target))
                if features:
                    mlflow.log_metric("train_features_dim", len(features[0]))

                # Обучаем на новых данных
                model.fit(features, target)

                # Метрика на train
                y_pred = model.predict(features)
                acc = accuracy_score(target, y_pred)
                mlflow.log_metric("train_accuracy", acc)

                # Перезаписываем старый файл модели
                joblib.dump(model, save_path)
                upload_model_to_s3(save_path, model_id)

                # Логируем модель в MLflow
                mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.error("MLflow tracking failed in retrain_model, retraining without MLflow: %s", e)
            model.fit(features, target)
            joblib.dump(model, save_path)
            upload_model_to_s3(save_path, model_id)
    else:
        # MLflow недоступен, тогда просто переобучаем
        model.fit(features, target)
        joblib.dump(model, save_path)
        upload_model_to_s3(save_path, model_id)