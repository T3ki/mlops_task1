import joblib
import uuid
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List, Dict, Any
import pandas as pd

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
    Обучает модель и сохраняет ее.

    :param model_name: Имя модели из AVAILABLE_MODELS.
    :param hyperparams: Гиперпараметры для модели.
    :param features: Признаки для обучения.
    :param target: Целевая переменная.
    :return: Уникальный ID обученной модели.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' is not available.")

    model_class = AVAILABLE_MODELS[model_name]
    
    # Создаем экземпляр модели с переданными гиперпараметрами
    model = model_class(**hyperparams)
    
    # Обучаем модель
    model.fit(features, target)
    
    # Генерируем уникальный ID и путь для сохранения
    model_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
    save_path = MODELS_DIR / f"{model_id}.joblib"
    
    # Сохраняем модель
    joblib.dump(model, save_path)
    
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
        raise FileNotFoundError(f"Model with id '{model_id}' not found.")

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

    if not model_path.exists():
        raise FileNotFoundError(f"Model with id '{model_id}' not found.")

    os.remove(model_path)

def retrain_model(
    model_id: str,
    hyperparams: Dict[str, Any],
    features: List[List[float]],
    target: List[int]
) -> None:
    """
    Переобучает существующую модель на новых данных и/или с новыми гиперпараметрами.

    :param model_id: ID модели для переобучения.
    :param hyperparams: Новые гиперпараметры для модели.
    :param features: Новые признаки для обучения.
    :param target: Новая целевая переменная.
    """
    save_path = MODELS_DIR / f"{model_id}.joblib"
    if not save_path.exists():
        raise FileNotFoundError(f"Model with id '{model_id}' not found for retraining.")

    # Извлекаем имя класса модели из ID
    # Например, из "logreg_a1b2c3d4" получаем "logreg"
    try:
        model_name = model_id.split('_')[0]
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type '{model_name}' in model_id.")
    except IndexError:
        raise ValueError(f"Invalid model_id format: '{model_id}'. Expected 'name_uuid'.")

    model_class = AVAILABLE_MODELS[model_name]
    
    # Создаем новый экземпляр модели с новыми гиперпараметрами
    model = model_class(**hyperparams)
    
    # Обучаем на новых данных
    model.fit(features, target)
    
    # Перезаписываем старый файл модели
    joblib.dump(model, save_path)