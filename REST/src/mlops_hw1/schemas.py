from pydantic import BaseModel
from typing import List, Dict, Any

class TrainRequest(BaseModel):
    """
    Модель запроса для обучения модели.
    """
    model_name: str
    hyperparams: Dict[str, Any] = {}  # Словарь с гиперпараметрами
    # Данные для обучения: список списков признаков и список меток
    features: List[List[float]]
    target: List[int]

    class Config:
        schema_extra = {
            "example": {
                "model_name": "logreg",
                "hyperparams": {"C": 1.0, "penalty": "l2"},
                "features": [[1.0, 2.5], [0.5, 1.8], [2.2, 0.9]],
                "target": [1, 0, 1]
            }
        }

class PredictRequest(BaseModel):
    """
    Модель запроса для предсказания.
    """
    model_id: str
    features: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "model_id": "logreg_a1b2c3d4",
                "features": [[0.8, 1.5], [2.0, 0.7]]
            }
        }

class RetrainRequest(BaseModel):
    """
    Модель запроса для переобучения модели.
    Не включает model_name, так как он определяется из model_id.
    """
    hyperparams: Dict[str, Any] = {}
    features: List[List[float]]
    target: List[int]

    class Config:
        schema_extra = {
            "example": {
                "hyperparams": {"C": 1.5}, 
                "features": [[1.1, 2.6], [0.6, 1.9], [2.3, 1.0]],
                "target": [1, 0, 1]
            }
        }