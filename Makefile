.PHONY: help setup-dev clean test lint type-check security-scan build deploy-local deploy-staging deploy-production

# Default target
help:
	@echo "Dynamic Pricing with Multimodal Geospatial Intelligence"
	@echo "Available targets:"
	@echo "  setup-dev          Setup development environment"
	@echo "  clean              Clean build artifacts and cache"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  test-performance   Run performance tests"
	@echo "  lint               Run code linting"
	@echo "  type-check         Run type checking"
	@echo "  security-scan      Run security scanning"
	@echo "  build              Build all Docker images"
	@echo "  deploy-local       Deploy to local development environment"
	@echo "  deploy-staging     Deploy to staging environment"
	@echo "  deploy-production  Deploy to production environment"

# Development Environment Setup
setup-dev:
	@echo "Setting up development environment..."
	python -m venv venv
	./venv/bin/pip install --upgrade pip setuptools wheel
	./venv/bin/pip install --use-pep517 -r requirements-dev.txt
	./venv/bin/pre-commit install
	@echo "Development environment setup complete!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	docker system prune -f
	@echo "Clean complete!"

# Testing
test: test-unit test-integration
	@echo "All tests completed!"

test-unit:
	@echo "Running unit tests..."
	pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --maxfail=1

test-performance:
	@echo "Running performance tests..."
	pytest tests/performance/ -v --benchmark-only

# Code Quality
lint:
	@echo "Running code linting..."
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	@echo "Note: pylint removed due to dependency conflicts. Install separately if needed."

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

type-check:
	@echo "Running type checking..."
	mypy src/

security-scan:
	@echo "Running security scanning..."
	bandit -r src/
	@echo "Note: safety and semgrep removed due to dependency conflicts. Install separately if needed."

# Docker Build
build:
	@echo "Building Docker images..."
	docker build -f infrastructure/docker/Dockerfile.processor -t pricing-image-processor:latest .
	docker build -f infrastructure/docker/Dockerfile.api -t pricing-api-gateway:latest .
	docker build -f infrastructure/docker/Dockerfile.pricing -t pricing-engine:latest .
	docker build -f infrastructure/docker/Dockerfile.ingestion -t data-ingestion:latest .
	docker build -f infrastructure/docker/Dockerfile.stream -t stream-processor:latest .

# Local Development
deploy-local:
	@echo "Deploying to local development environment..."
	docker-compose -f docker-compose.yml up -d
	@echo "Local deployment complete! Services available at:"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

stop-local:
	@echo "Stopping local development environment..."
	docker-compose -f docker-compose.yml down

# Infrastructure
terraform-init:
	@echo "Initializing Terraform..."
	cd infrastructure/terraform && terraform init

terraform-plan:
	@echo "Planning Terraform deployment..."
	cd infrastructure/terraform && terraform plan

terraform-apply:
	@echo "Applying Terraform configuration..."
	cd infrastructure/terraform && terraform apply

terraform-destroy:
	@echo "Destroying Terraform infrastructure..."
	cd infrastructure/terraform && terraform destroy

# Kubernetes Deployment
k8s-deploy-staging:
	@echo "Deploying to Kubernetes staging..."
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/staging/ -n pricing-staging

k8s-deploy-production:
	@echo "Deploying to Kubernetes production..."
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/production/ -n pricing-production

# BigQuery Setup
bigquery-setup:
	@echo "Setting up BigQuery datasets and tables..."
	python scripts/setup/create_datasets.py
	python scripts/data/data_migration.py

# ML Model Management
train-models:
	@echo "Training ML models..."
	python scripts/ml/train_models.py

deploy-models:
	@echo "Deploying ML models to Vertex AI..."
	python scripts/ml/model_deployment.py

# Data Management
load-sample-data:
	@echo "Loading sample data..."
	python scripts/data/sample_data_generator.py
	python scripts/data/data_migration.py

validate-data:
	@echo "Validating data quality..."
	python scripts/data/data_validation.py

# Monitoring
setup-monitoring:
	@echo "Setting up monitoring stack..."
	kubectl apply -f infrastructure/kubernetes/monitoring/

# Environment-specific deployments
deploy-staging: build terraform-apply k8s-deploy-staging bigquery-setup
	@echo "Staging deployment complete!"

deploy-production: build terraform-apply k8s-deploy-production
	@echo "Production deployment complete!"

# Health Checks
health-check:
	@echo "Running system health checks..."
	python scripts/operations/health_check.py

# Backup and Recovery
backup-data:
	@echo "Backing up data..."
	bash scripts/operations/backup_data.sh

# Documentation
docs-build:
	@echo "Building documentation..."
	cd docs && mkdocs build

docs-serve:
	@echo "Serving documentation locally..."
	cd docs && mkdocs serve

# CI/CD Support
ci-test: lint type-check security-scan test
	@echo "CI pipeline tests completed!"

ci-build: clean build
	@echo "CI build completed!"

# Development Utilities
logs-local:
	@echo "Showing local development logs..."
	docker-compose logs -f

logs-staging:
	@echo "Showing staging logs..."
	kubectl logs -f -l app=pricing-engine -n pricing-staging

logs-production:
	@echo "Showing production logs..."
	kubectl logs -f -l app=pricing-engine -n pricing-production

# Database Operations
db-migrate:
	@echo "Running database migrations..."
	python scripts/data/data_migration.py

db-seed:
	@echo "Seeding database with initial data..."
	python scripts/setup/seed_database.py

# Performance Monitoring
benchmark:
	@echo "Running performance benchmarks..."
	python tests/performance/benchmark_ml_models.py
	locust -f tests/performance/load_test_pricing_api.py --headless -u 100 -r 10 -t 60s

# Cleanup Operations
cleanup-resources:
	@echo "Cleaning up cloud resources..."
	bash scripts/operations/cleanup_resources.sh

# Version Management
version:
	@echo "Current version information:"
	@git describe --tags --always --dirty
	@echo "Last commit: $(shell git log -1 --format='%h %s')"
