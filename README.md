# mlops_task1

Состав команды: Останкова Софья, Басов Даниил, Никифорова Елизавета

**Работа состояла из 3 блоков:**
1. REST API
2. Реализация gRPC процедур 
3. Создание дашборда на streamlit 

### REST API

Был создан REST API сервис для управления ML моделями. В структуре сервиса несколько слоев: 
1. FastAPI эндпоинты в main.py
2. Модуль для работы с ML моделями (обучение, предсказание, управление) в models.py (доступные модели - LogisticRegression и RandomForestClassifier)
3. Схемы данных для валидации запросов с примерами в schemas.py

Для запуска через терминал: 
1. git clone https://github.com/T3ki/mlops_task1 
2. cd mlops_task1
3. git checkout dev
4. poetry run uvicorn src.mlops_hw1.main:app --host localhost --port 8000 --reload
   переходим по ссылке, проверяем, что работает http://localhost:8000/status

### Реализация gRPC процедур

1. mlops_hw1.proto  -  описание gRPC-сервиса
2. mlops_hw1_pb2.py - автоматически сгенерировано из .proto
3. mlops_hw1_pb2_grpc.py - автоматически сгенерировано из .proto
4. grpc_server.py - реализация сервера
5. grpc_client.py - тестовый клиент - проверяет все 6 методов подряд

Для запуска через терминал:
1. Запуск gRPC-сервера (в отдельном терминале)
   poetry run python -m src.mlops_hw1.grpc.grpc_server

2. Тестирование сервиса (в новом терминале)
   poetry run python -m src.mlops_hw1.grpc.grpc_client

Ожидаемый вывод
Status: ok
Available models: ['logreg', 'rf']
Trained model ID: logreg
Predictions: [0]
Retrain message: Model retrained successfully
Delete message: Model deleted successfully

### Создание дашборда на streamlit

Дашборд опирается на REST API сервис. Код в файлике dashboard.py

После запуска API в отдельном терминале: 

1. cd mlops_task1
2. git checkout dev
3. poetry run streamlit run dashboard.py



# mlops_task2

**Что добавлено:**

- Хранение обученных моделей в Minio (S3) через src/mlops_hw1/storage.py
- Хранение и версионирование обучающих датасетов через DVC с remote в Minio
- Docker-образ сервиса (через Dockerfile)
- Запуск Minio и сервиса через docker-compose.yml
- Трекинг обученных моделей в mlflow

**Основные файлы HW2:**

-- Dockerfile

-- docker-compose.yml

-- .dvc/                     # конфигурация DVC (remote на Minio)

-- data/

----  train_datasets/         # *.csv.dvc (сырые .csv хранятся в Minio через DVC)

--src/mlops_hw1/

----  models.py               # логика моделей + работа с S3 и DVC

----  storage.py              # работа с S3 (Minio) для моделей

----  dvc_utils.py            # сохранение датасетов и вызовы DVC (dvc add + dvc push)

---- tracking.py              # Инициализация MLflow

### Запуск через docker-compose

В корне репозитория:

```
docker compose up --build
```

Поднимаются сервисы:

- Minio
   - S3 API: http://localhost:9000
   - Web UI: http://localhost:9001
- Приложение
   - REST API (FastAPI): http://localhost:8000
   - gRPC: порт 50051
- MLflow: http://localhost:5000

### Настройка Minio

1. Открыть в браузере http://localhost:9001.
   - Логин: minioadmin
   - Пароль: minioadmin123
2. Создать бакеты:
   - mlops-hw1-models — для обученных моделей;
   - mlops-hw1-dvc — для датасетов, которыми управляет DVC.
   - mlops-hw1-mlflow - для артефактов моделей из mlflow

### Как пользоваться сервисом

1. Открыть Swagger: http://localhost:8000/docs.
2. Проверить статус:
      - GET /status
3. Обучить модель:
      - POST /train
      - В ответе приходит model_id
      - Проверить в Minio:
            что в бакете mlops-hw1-models появился models/<model_id>.joblib
            что в бакете mlops-hw1-dvc появились новые объекты от DVC
      - Проверить, что в mlflow в эксперименте mlops_hw1 появилась информация о модели и сама модель
4. Сделать предсказания: 
      - POST /predict с тем же model_id и features
5. Управление моделями:
      - DELETE /models/{model_id} — удаляет модель локально и из Minio
      - POST /models/{model_id}/retrain — переобучает модель на новых данных, обновляет версию в Minio, добавляет новую модель в MLflow
