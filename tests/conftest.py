"""
Configuração global de testes — Fixtures compartilhadas.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


@pytest.fixture
def sample_patient_data() -> dict[str, Any]:
    """Dados de paciente sintético para testes."""
    return {
        "id": "P001",
        "nome": "João Silva",
        "idade": 65,
        "sexo": "M",
        "queixa_principal": "Dor torácica intensa",
        "diagnosticos": ["Hipertensão", "Diabetes Tipo 2"],
        "medicamentos_em_uso": [
            {"nome": "Metformina", "dose": "850mg", "frequencia": "2x/dia"},
            {"nome": "Losartana", "dose": "50mg", "frequencia": "1x/dia"},
        ],
        "alergias": ["Penicilina"],
        "sinais_vitais": {
            "pa_sistolica": 180,
            "pa_diastolica": 110,
            "fc": 105,
            "fr": 22,
            "temperatura": 37.2,
            "spo2": 93,
        },
        "exames": [
            {"nome": "Hemograma", "status": "concluido", "resultado": "Normal"},
            {"nome": "ECG", "status": "pendente", "resultado": None},
            {"nome": "Troponina", "status": "pendente", "resultado": None},
        ],
    }


@pytest.fixture
def sample_patient_regular() -> dict[str, Any]:
    """Paciente com dados regulares (não-críticos)."""
    return {
        "id": "P002",
        "nome": "Maria Souza",
        "idade": 35,
        "sexo": "F",
        "queixa_principal": "Cefaleia leve",
        "diagnosticos": [],
        "medicamentos_em_uso": [],
        "alergias": [],
        "sinais_vitais": {
            "pa_sistolica": 120,
            "pa_diastolica": 80,
            "fc": 72,
            "fr": 16,
            "temperatura": 36.5,
            "spo2": 98,
        },
        "exames": [],
    }


@pytest.fixture
def sample_pubmedqa_entry() -> dict[str, Any]:
    """Entrada PubMedQA processada."""
    return {
        "instruction": "Com base no contexto fornecido, a metformina é eficaz para diabetes tipo 2?",
        "input": "Estudo randomizado com 500 pacientes...",
        "output": "Sim. A metformina demonstrou eficácia significativa...",
        "label": "sim",
    }


@pytest.fixture
def temp_jsonl_file(sample_pubmedqa_entry) -> Generator[str, None, None]:
    """Cria arquivo JSONL temporário."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as f:
        for _ in range(10):
            f.write(json.dumps(sample_pubmedqa_entry, ensure_ascii=False) + "\n")
        path = f.name

    yield path
    os.unlink(path)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Cria diretório temporário."""
    with tempfile.TemporaryDirectory() as d:
        yield d
