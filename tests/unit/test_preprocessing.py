"""
Testes unitários — Pré-processamento de dados.
"""

import json
import os
import tempfile

import pytest

from medical_assistant.evaluation.metrics import extract_answer_label


class TestPubMedQAProcessing:
    """Testes para o formato de saída do processamento PubMedQA."""

    def test_alpaca_format_fields(self, sample_pubmedqa_entry):
        """Verifica que a entrada processada tem os campos Alpaca esperados."""
        assert "instruction" in sample_pubmedqa_entry
        assert "input" in sample_pubmedqa_entry
        assert "output" in sample_pubmedqa_entry

    def test_label_mapping(self):
        """Verifica mapeamento de labels."""
        assert extract_answer_label("Sim, os dados mostram...") == "sim"
        assert extract_answer_label("Não, o estudo não confirma...") == "não"
        assert extract_answer_label("Talvez, são necessários mais estudos.") == "talvez"


class TestFormatConverter:
    """Testes para conversão de formatos."""

    def test_jsonl_roundtrip(self):
        """Verifica que dados podem ser salvos e carregados em JSONL."""
        data = [
            {"instruction": "Q1", "input": "C1", "output": "A1"},
            {"instruction": "Q2", "input": "C2", "output": "A2"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            path = f.name

        try:
            loaded = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))

            assert len(loaded) == 2
            assert loaded[0]["instruction"] == "Q1"
            assert loaded[1]["output"] == "A2"
        finally:
            os.unlink(path)
