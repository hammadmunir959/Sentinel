.PHONY: install install-dev preprocess serve test lint clean

# Install production dependencies
install:
	pip install -r requirements.txt

# Install all dependencies (including dev)
install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# Run preprocessing pipeline
preprocess:
	python src/preprocess.py

# Start API server (development)
serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start API server (production)
serve-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run linting
lint:
	ruff check src/ tests/

# Fix linting issues
lint-fix:
	ruff check src/ tests/ --fix

# Type checking
typecheck:
	mypy src/

# Build Docker image
docker-build:
	docker build -t sentinel .

# Run Docker container
docker-run:
	docker run -p 8000:8000 sentinel

# Generate drift report
drift-report:
	python src/monitor.py

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
