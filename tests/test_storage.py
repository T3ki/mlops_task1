import pytest
import boto3
from moto import mock_aws
from mlops_hw1.storage import upload_model_to_s3, download_model_from_s3, S3_BUCKET_MODELS
from sklearn.linear_model import LogisticRegression
import joblib
import os
from mlops_hw1 import storage

@pytest.fixture
def mock_s3(mocker):
    with mock_aws():
        mock_client = boto3.client("s3", region_name="us-east-1")
        mock_client.create_bucket(Bucket=S3_BUCKET_MODELS)
        mocker.patch.object(storage, '_s3_client', mock_client)

        yield mock_client

        
        
def test_upload_and_download_model(mock_s3):
    model = LogisticRegression()
    model_id = "logreg_test"
    local_path = f"models_storage/{model_id}.joblib"
    os.makedirs("models_storage", exist_ok=True)
    joblib.dump(model, local_path)

    upload_model_to_s3(local_path, model_id) 

    objects = mock_s3.list_objects_v2(Bucket=S3_BUCKET_MODELS)
    assert objects.get("KeyCount", 0) == 1
    assert objects["Contents"][0]["Key"] == f"models/{model_id}.joblib"

    os.remove(local_path)

    download_model_from_s3(local_path, model_id)

    assert os.path.exists(local_path)

    os.remove(local_path)
    
def test_s3_bucket_name():
    assert S3_BUCKET_MODELS == "mlops-hw1-models"