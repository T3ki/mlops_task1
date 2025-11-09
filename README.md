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
