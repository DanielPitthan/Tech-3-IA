"""
Testes unitários — Pré-processamento de dados.
"""

import json
import os
import tempfile

import pytest

from medical_assistant.data.preprocessing.dataset_splitter import load_test_ids, split_dataset
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


class TestDatasetSplitter:
    """Testes para split com ground truth fixo."""

    def test_load_test_ids(self):
        """Carrega IDs de teste a partir de JSON de ground truth."""
        payload = {"1001": "yes", "1002": "no", "1003": "maybe"}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(payload, f)
            path = f.name

        try:
            test_ids = load_test_ids(path)
            assert test_ids == {"1001", "1002", "1003"}
        finally:
            os.unlink(path)

    def test_split_respects_ground_truth_ids(self):
        """Garante que IDs do ground truth entrem no conjunto de teste."""
        samples = [
            {"pmid": "1001", "label": "yes", "instruction": "i", "input": "x", "output": "o"},
            {"pmid": "1002", "label": "no", "instruction": "i", "input": "x", "output": "o"},
            {"pmid": "1003", "label": "maybe", "instruction": "i", "input": "x", "output": "o"},
            {"pmid": "1004", "label": "yes", "instruction": "i", "input": "x", "output": "o"},
            {"pmid": "1005", "label": "no", "instruction": "i", "input": "x", "output": "o"},
            {"pmid": "1006", "label": "maybe", "instruction": "i", "input": "x", "output": "o"},
        ]

        splits = split_dataset(samples, test_ids={"1002", "1005"}, val_ratio=0.5, seed=123)

        test_pmids = {item["pmid"] for item in splits["test"]}
        train_pmids = {item["pmid"] for item in splits["train"]}
        val_pmids = {item["pmid"] for item in splits["val"]}

        assert test_pmids == {"1002", "1005"}
        assert test_pmids.isdisjoint(train_pmids)
        assert test_pmids.isdisjoint(val_pmids)
