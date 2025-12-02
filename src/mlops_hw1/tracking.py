import logging
import os

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:  # на всякий случай для локального запуска без mlflow
    mlflow = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops_hw1")


def init_mlflow() -> bool:
    """Инициализирует MLflow, возвращает True, если трекинг включен."""
    if not MLFLOW_AVAILABLE:
        return False

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        return True
    except Exception as e:
        logger.error("Failed to init MLflow: %s", e)
        return False