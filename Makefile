.PHONY: help install test lint format clean docker dev prod

help:
	@echo "Rugby IA — Makefile commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install dev dependencies (tests, linting)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-cov         Run tests with coverage"
	@echo "  make test-fast        Run fast tests (skip integration)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting (flake8, pylint)"
	@echo "  make format           Format code with Black"
	@echo "  make format-check     Check formatting without changes"
	@echo ""
	@echo "Development:"
	@echo "  make dev              Start dev servers (API + Dashboard)"
	@echo "  make api              Start FastAPI server"
	@echo "  make dashboard        Start Streamlit dashboard"
	@echo "  make mlflow           Start MLflow UI"
	@echo ""
	@echo "Data & Training:"
	@echo "  make download-data    Download Roboflow dataset"
	@echo "  make train            Train YOLOv8 model (CPU)"
	@echo "  make train-quick      Train quick test (3 epochs, CPU)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-compose   Run with docker-compose"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache and temp files"
	@echo "  make clean-all        Remove cache, models, and runs"

# --- Setup ---
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov pytest-benchmark
	pip install pylint flake8 black isort bandit safety
	pip install radon  # Code metrics

# --- Testing ---
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not integration"

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

test-quick: test-fast

# --- Code Quality ---
lint:
	@echo "=== Flake8 ===" && \
	flake8 src tests scripts --max-line-length=120 || true && \
	echo "\n=== Pylint ===" && \
	pylint src --disable=all --enable=E,F --exit-zero --max-line-length=120 || true

format:
	black src tests scripts --line-length=120
	isort src tests scripts

format-check:
	black --check src tests scripts --line-length=120
	isort --check-only src tests scripts

# --- Development ---
dev:
	@echo "Starting API (port 8000) and Dashboard (port 8501)..."
	@echo "Open: http://localhost:8501"
	concurrently "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload" \
	             "streamlit run dashboard/app.py --server.port 8501"

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

mlflow:
	mlflow ui --port 5000

# --- Data & Training ---
download-data:
	@read -p "Enter Roboflow API key: " API_KEY && \
	python scripts/download_roboflow_dataset.py --api-key $$API_KEY

train:
	python scripts/finetune_yolo_rugby.py \
		--data data/roboflow/merged/data.yaml \
		--epochs 15 \
		--device cpu \
		--batch 4

train-quick:
	python scripts/finetune_yolo_rugby.py \
		--data data/roboflow/merged/data.yaml \
		--epochs 3 \
		--device cpu \
		--batch 4 \
		--imgsz 640

train-val:
	python scripts/finetune_yolo_rugby.py \
		--val-only \
		--data data/roboflow/merged/data.yaml \
		--device cpu

# --- Docker ---
docker-build:
	docker build -t rugby-ia:latest .

docker-run: docker-build
	docker run -p 8000:8000 -p 8501:8501 rugby-ia:latest

docker-compose:
	docker-compose up --build

# --- Cleanup ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage *.log

clean-all: clean
	rm -rf data/models/*.pt
	rm -rf runs/
	rm -rf mlruns/
	rm -rf dist/ build/ *.egg-info
	@echo "Cleaned all build artifacts and trained models"
