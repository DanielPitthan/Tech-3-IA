.PHONY: install lint format test test-unit test-integration preprocess train evaluate chat clean help

# ──────────────────────────────────────────────
# MedAssist — Makefile
# ──────────────────────────────────────────────

help: ## Exibir ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependências (venv + pip)
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

lint: ## Executar linting (ruff + mypy)
	python -m ruff check medical_assistant/ tests/
	python -m mypy medical_assistant/ --ignore-missing-imports

format: ## Formatar código (ruff)
	python -m ruff format medical_assistant/ tests/

test: ## Executar todos os testes
	python -m pytest tests/ -v --tb=short

test-unit: ## Executar testes unitários
	python -m pytest tests/unit/ -v --tb=short

test-integration: ## Executar testes de integração
	python -m pytest tests/integration/ -v --tb=short

preprocess: ## Pré-processar datasets
	python app.py --preprocess

index: ## Indexar base de conhecimento no ChromaDB
	python app.py --index-knowledge

train: ## Executar fine-tuning QLoRA
	python app.py --finetune

generate-patients: ## Gerar pacientes sintéticos
	python app.py --generate-patients --count 20

chat: ## Iniciar chat interativo
	python app.py --chat

evaluate: ## Executar benchmark
	python app.py --evaluate benchmark

judge: ## Executar LLM-as-judge
	python app.py --evaluate judge --max-samples 50

report: ## Gerar relatório de avaliação
	python app.py --evaluate report

run-all: ## Pipeline completo (interativo)
	python app.py --all

clean: ## Limpar artefatos
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

clean-data: ## Limpar dados processados
	rm -rf data/processed/ data/patients/ chroma_db/ evaluation_results/

all: install preprocess index train evaluate ## Pipeline completo (sequencial)
