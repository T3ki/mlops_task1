import grpc
from concurrent import futures
import logging
from pathlib import Path

from . import mlops_hw1_pb2
from . import mlops_hw1_pb2_grpc

from ..models import (
    get_available_models,
    train_model,
    predict_with_model,
    delete_model,
    retrain_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLServiceServicer(mlops_hw1_pb2_grpc.MLServiceServicer):
    def GetStatus(self, request, context):
        logger.info("gRPC: Check status ok")
        return mlops_hw1_pb2.StatusResponse(status="ok")

    def ListModels(self, request, context):
        models = get_available_models()
        logger.info(f"gRPC: Returned available models: {models}")
        return mlops_hw1_pb2.ListModelsResponse(models=models)

    def Train(self, request, context):
        try:
            features = [[val for val in row.values] for row in request.features]
            target = [int(x) for x in request.target]
            hyperparams = {k: float(v) if k == "C" else v 
               for k, v in request.hyperparams.items()}  # map to dict

            model_id = train_model(
                model_name=request.model_name,
                hyperparams=hyperparams,
                features=features,
                target=target
            )
            logger.info(f"gRPC: Trained model ID: {model_id}")
            return mlops_hw1_pb2.TrainResponse(model_id=model_id, message="Model trained successfully")
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return mlops_hw1_pb2.TrainResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error")
            return mlops_hw1_pb2.TrainResponse()

    def Predict(self, request, context):
        try:
            features = [[val for val in row.values] for row in request.features]
            predictions = predict_with_model(model_id=request.model_id, features=features)
            logger.info(f"gRPC: Prediction successful for model: {request.model_id}")
            return mlops_hw1_pb2.PredictResponse(predictions=predictions)
        except FileNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return mlops_hw1_pb2.PredictResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error")
            return mlops_hw1_pb2.PredictResponse()

    def DeleteModel(self, request, context):
        try:
            delete_model(model_id=request.model_id)
            logger.info(f"gRPC: Deleted model: {request.model_id}")
            return mlops_hw1_pb2.DeleteResponse(message="Model deleted successfully")
        except FileNotFoundError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return mlops_hw1_pb2.DeleteResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error")
            return mlops_hw1_pb2.DeleteResponse()

    def Retrain(self, request, context):
        try:
            features = [[val for val in row.values] for row in request.features]
            target = [int(x) for x in request.target]
            hyperparams = {k: float(v) if k == "C" else v 
               for k, v in request.hyperparams.items()}

            retrain_model(
                model_id=request.model_id,
                hyperparams=hyperparams,
                features=features,
                target=target
            )
            logger.info(f"gRPC: Retrained model: {request.model_id}")
            return mlops_hw1_pb2.RetrainResponse(message="Retrained successfully")
        except (FileNotFoundError, ValueError) as e:
            code = grpc.StatusCode.NOT_FOUND if isinstance(e, FileNotFoundError) else grpc.StatusCode.INVALID_ARGUMENT
            context.set_code(code)
            context.set_details(str(e))
            return mlops_hw1_pb2.RetrainResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal error")
            return mlops_hw1_pb2.RetrainResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mlops_hw1_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port('[::]:50051') 
    server.start()
    logger.info("gRPC старт на порте 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()