FROM python:3.11-slim

# Системные зависимости для сборки некоторых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Настройки Poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Устанавливаем Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Ставим только зависимости без установки самого проекта
RUN poetry install --no-root --no-dev

# Копируем остальной код проекта
COPY . .

# Теперь, когда код на месте, устанавливаем и сам проект как пакет
RUN poetry install --no-dev

# Логи без буферизации
ENV PYTHONUNBUFFERED=1

# Порты: 8000 - FastAPI, 50051 - gRPC
EXPOSE 8000 50051

# Запускаем gRPC сервер и FastAPI в одном контейнере
CMD ["bash", "-c", "python -m mlops_hw1.grpc.grpc_server & uvicorn mlops_hw1.main:app --host 0.0.0.0 --port 8000"]