import grpc
import sys
from pathlib import Path

# Импорт сгенерированных файлов
from . import mlops_hw1_pb2
from . import mlops_hw1_pb2_grpc
def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = mlops_hw1_pb2_grpc.MLServiceStub(channel)

        # Тест статуса
        response = stub.GetStatus(mlops_hw1_pb2.Empty())
        print("Status:", response.status)

        # Тест списка моделей
        response = stub.ListModels(mlops_hw1_pb2.Empty())
        print("Available models:", response.models)

        # Тест обучения
        train_req = mlops_hw1_pb2.TrainRequest(
            model_name="logreg",
            hyperparams={"C": "1.0", "penalty": "l2"},
            features=[mlops_hw1_pb2.FeaturesRow(values=[1.0, 2.5]), mlops_hw1_pb2.FeaturesRow(values=[0.5, 1.8])],
            target=[1,0]
        )
        response = stub.Train(train_req)
        model_id = response.model_id
        print("Trained model ID:", model_id)

        # Тест предсказания
        predict_req = mlops_hw1_pb2.PredictRequest(
            model_id=model_id,
            features=[mlops_hw1_pb2.FeaturesRow(values=[0.8, 1.5])]
        )
        response = stub.Predict(predict_req)
        print("Predictions:", response.predictions)

        # Тест переобучения
        retrain_req = mlops_hw1_pb2.RetrainRequest(
            model_id=model_id,
            hyperparams={"C": "1.5"},
            features=[
                mlops_hw1_pb2.FeaturesRow(values=[1.1, 2.6]),
                mlops_hw1_pb2.FeaturesRow(values=[0.3, 1.2])
            ],
            target=[1, 0]
        )
        response = stub.Retrain(retrain_req)
        print("Retrain message:", response.message)

        # Тест удаления
        delete_req = mlops_hw1_pb2.DeleteRequest(model_id=model_id)
        response = stub.DeleteModel(delete_req)
        print("Delete message:", response.message)

if __name__ == '__main__':
    run()