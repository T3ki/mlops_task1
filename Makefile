IMAGE_NAME = t3ki1/mlops_task
VERSION = latest

build-push:
	docker build -t $(IMAGE_NAME):$(VERSION) .
	docker push $(IMAGE_NAME):$(VERSION)

test:
	poetry run pytest tests/ -v

lint:
	poetry run black --check src/ tests/
	poetry run flake8 src/ tests/

format:
	poetry run black src/ tests/