"""
Testes unitários — Métricas de avaliação.
"""

import pytest

from medical_assistant.evaluation.metrics import (
    compute_metrics,
    extract_answer_label,
    compute_exact_match,
    compute_token_f1,
    compute_average_token_f1,
)


class TestComputeMetrics:
    """Testes para compute_metrics."""

    def test_perfect_predictions(self):
        y_true = ["sim", "não", "talvez", "sim"]
        y_pred = ["sim", "não", "talvez", "sim"]
        result = compute_metrics(y_true, y_pred)
        assert result.accuracy == 1.0
        assert result.f1_macro == 1.0

    def test_all_wrong(self):
        y_true = ["sim", "sim", "sim"]
        y_pred = ["não", "não", "não"]
        result = compute_metrics(y_true, y_pred)
        assert result.accuracy == 0.0

    def test_partial(self):
        y_true = ["sim", "não", "sim", "não"]
        y_pred = ["sim", "não", "não", "não"]
        result = compute_metrics(y_true, y_pred)
        assert 0.0 < result.accuracy < 1.0
        assert result.correct == 3

    def test_empty_lists(self):
        result = compute_metrics([], [])
        assert result.accuracy == 0.0
        assert result.total_samples == 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            compute_metrics(["sim", "não"], ["sim"])

    def test_confusion_matrix_shape(self):
        y_true = ["sim", "não", "talvez"]
        y_pred = ["sim", "não", "sim"]
        result = compute_metrics(y_true, y_pred, labels=["sim", "não", "talvez"])
        assert len(result.confusion_matrix) == 3
        assert len(result.confusion_matrix[0]) == 3


class TestExtractAnswerLabel:
    """Testes para extract_answer_label."""

    def test_sim(self):
        assert extract_answer_label("Sim, de acordo com o estudo...") == "sim"
        assert extract_answer_label("Yes, the study shows...") == "sim"

    def test_nao(self):
        assert extract_answer_label("Não, os dados não suportam...") == "não"
        assert extract_answer_label("No, the evidence suggests...") == "não"

    def test_talvez(self):
        assert extract_answer_label("Talvez, mais estudos são necessários.") == "talvez"
        assert extract_answer_label("Maybe, further research is needed.") == "talvez"

    def test_unknown(self):
        assert extract_answer_label("Os dados são inconclusivos.") == "desconhecido"

    def test_empty(self):
        assert extract_answer_label("") == "desconhecido"


class TestExactMatch:
    """Testes para compute_exact_match."""

    def test_all_match(self):
        refs = ["hello", "world"]
        preds = ["hello", "world"]
        assert compute_exact_match(refs, preds) == 1.0

    def test_none_match(self):
        refs = ["hello", "world"]
        preds = ["hi", "earth"]
        assert compute_exact_match(refs, preds) == 0.0

    def test_case_insensitive(self):
        refs = ["Hello"]
        preds = ["hello"]
        assert compute_exact_match(refs, preds) == 1.0

    def test_empty(self):
        assert compute_exact_match([], []) == 0.0


class TestTokenF1:
    """Testes para compute_token_f1."""

    def test_identical(self):
        assert compute_token_f1("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        f1 = compute_token_f1("hello world foo", "hello world bar")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        assert compute_token_f1("hello", "world") == 0.0

    def test_empty(self):
        assert compute_token_f1("", "") == 0.0

    def test_average(self):
        refs = ["the cat sat", "the dog ran"]
        preds = ["the cat sat", "the dog walked"]
        avg = compute_average_token_f1(refs, preds)
        assert 0.5 < avg <= 1.0
