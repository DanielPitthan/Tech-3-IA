.PHONY: install lint format test test-unit test-integration preprocess train evaluate chat clean help

# ──────────────────────────────────────────────
# MedAssist — Makefile
# ──────────────────────────────────────────────

help: ## Exibir ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependências
	pip install poetry
	poetry install

lint: ## Executar linting (ruff + mypy)
	poetry run ruff check medical_assistant/ tests/
	poetry run mypy medical_assistant/ --ignore-missing-imports

format: ## Formatar código (ruff)
	poetry run ruff format medical_assistant/ tests/

test: ## Executar todos os testes
	poetry run pytest tests/ -v --tb=short

test-unit: ## Executar testes unitários
	poetry run pytest tests/unit/ -v --tb=short

test-integration: ## Executar testes de integração
	poetry run pytest tests/integration/ -v --tb=short

preprocess: ## Pré-processar datasets
	poetry run medassist train preprocess

index: ## Indexar base de conhecimento no ChromaDB
	poetry run medassist train index-knowledge

train: ## Executar fine-tuning QLoRA
	poetry run medassist train finetune

generate-patients: ## Gerar pacientes sintéticos
	poetry run medassist train generate-patients --count 20

chat: ## Iniciar chat interativo
	poetry run medassist chat

evaluate: ## Executar benchmark
	poetry run medassist evaluate benchmark

judge: ## Executar LLM-as-judge
	poetry run medassist evaluate judge --max-samples 50

report: ## Gerar relatório de avaliação
	poetry run medassist evaluate report

clean: ## Limpar artefatos
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

clean-data: ## Limpar dados processados
	rm -rf data/processed/ data/patients/ chroma_db/ evaluation_results/

all: install preprocess index train evaluate ## Pipeline completo
