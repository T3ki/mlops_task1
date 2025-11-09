from fastapi import FastAPI, HTTPException, Body
import logging

from .models import get_available_models, train_model, predict_with_model, delete_model, retrain_model
from .schemas import TrainRequest, PredictRequest, RetrainRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps HW1 Service",
    description="A service for training and predicting with ML models.",
    version="0.1.0",
)

@app.get("/status", tags=["Monitoring"])
def get_status():
    """
    Проверяет статус сервиса.
    Возвращает 'ok', если сервис работает.
    """
    logger.info("Checked status: ok")
    return {"status": "ok"}

@app.get("/models", tags=["Models"])
def list_available_models():
    """
    Возвращает список доступных для обучения классов моделей.
    """
    available_models = get_available_models()
    logger.info(f"Returned available models: {available_models}")
    return {"models": available_models}

@app.post("/train", tags=["Models"])
def train_new_model(
    request: TrainRequest = Body(

        example=TrainRequest.Config.schema_extra["example"] 
    )
):
    """
    Обучает новую ML-модель.
    Принимает название модели, гиперпараметры и данные для обучения.
    Возвращает ID обученной модели.
    """
    try:
        logger.info(f"Starting training for model: {request.model_name}")
        model_id = train_model(
            model_name=request.model_name,
            hyperparams=request.hyperparams,
            features=request.features,
            target=request.target
        )
        logger.info(f"Finished training. Model ID: {model_id}")
        return {"message": "Model trained successfully", "model_id": model_id}
    except ValueError as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    

@app.post("/predict", tags=["Models"])
def predict_with_trained_model(
    request: PredictRequest = Body(
        ...,
        example=PredictRequest.Config.schema_extra["example"]
    )
):
    """
    Делает предсказание с помощью обученной модели.
    Принимает ID модели и данные для предсказания.
    Возвращает список предсказаний.
    """
    try:
        logger.info(f"Making prediction with model: {request.model_id}")
        predictions = predict_with_model(
            model_id=request.model_id,
            features=request.features
        )
        logger.info(f"Prediction successful for model: {request.model_id}")
        return {"predictions": predictions}
    except FileNotFoundError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
@app.delete("/models/{model_id}", tags=["Models"])
def remove_model(model_id: str):
    """
    Удаляет обученную модель.
    Принимает ID модели в пути URL.
    """
    try:
        logger.info(f"Attempting to delete model: {model_id}")
        delete_model(model_id)
        logger.info(f"Successfully deleted model: {model_id}")
        return {"message": "Model deleted successfully", "model_id": model_id}
    except FileNotFoundError as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during deletion: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
@app.post("/models/{model_id}/retrain", tags=["Models"])
def retrain_existing_model(
    model_id: str,
    request: RetrainRequest = Body(
        ...,
        example=RetrainRequest.Config.schema_extra["example"]
    )
):
    """
    Переобучает существующую модель.
    Принимает ID модели, новые гиперпараметры и новые данные.
    """
    try:
        logger.info(f"Starting retraining for model: {model_id}")
        retrain_model(
            model_id=model_id,
            hyperparams=request.hyperparams,
            features=request.features,
            target=request.target
        )
        logger.info(f"Finished retraining for model: {model_id}")
        return {"message": "Model retrained successfully", "model_id": model_id}
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Retraining failed: {e}")
        # Используем 404 для ненайденной модели и 400 для неверных данных
        status_code = 404 if isinstance(e, FileNotFoundError) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred during retraining: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")