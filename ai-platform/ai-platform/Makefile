.PHONY: install train run test eval lint demo docker
install:
	pip install -r requirements-dev.txt
train:
	python -m app.ml.train
run:
	uvicorn app.main:app --reload --port 8000
test:
	pytest -q
eval:
	python -m app.evaluation.run
lint:
	ruff check .
demo:
	python -m scripts.demo
docker:
	docker compose up --build
