import pytest
from mlops_hw1.models import train_model, predict_with_model
import os

def test_train_and_predict_logreg(mocker):
    # мокаются внешне зависимости
    mocker.patch('mlops_hw1.models.upload_model_to_s3')  # мок S3
    mocker.patch('mlops_hw1.models.save_dataset_with_dvc')  # мок DVC
    mocker.patch('mlops_hw1.models.init_mlflow', return_value=False)  #отключаем MLflow

    features = [[1.0, 2.0], [3.0, 4.0], [2.0, 1.0], [4.0, 3.0]]
    target = [0, 1, 0, 1]
    hyperparams = {"C": 1.0, "max_iter": 100}   
    model_id = train_model(model_name="logreg", hyperparams=hyperparams, features=features, target=target)
    
    assert model_id.startswith("logreg_")
    predictions = predict_with_model(model_id, [[1.5, 2.5]])
    assert isinstance(predictions[0], int)
    
    os.remove(f"models_storage/{model_id}.joblib")