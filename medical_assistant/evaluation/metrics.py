"""
Métricas de avaliação — Accuracy, F1, Confusion Matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Resultado consolidado de avaliação."""

    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)
    classification_report: str = ""
    labels: list[str] = field(default_factory=list)
    total_samples: int = 0
    correct: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_weighted": round(self.f1_weighted, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "total_samples": self.total_samples,
            "correct": self.correct,
            "classification_report": self.classification_report,
        }

    def summary(self) -> str:
        return (
            f"=== Resultados da Avaliação ===\n"
            f"Total de amostras: {self.total_samples}\n"
            f"Corretas: {self.correct}\n"
            f"Accuracy: {self.accuracy:.4f}\n"
            f"F1 (macro): {self.f1_macro:.4f}\n"
            f"F1 (weighted): {self.f1_weighted:.4f}\n"
            f"Precision (macro): {self.precision_macro:.4f}\n"
            f"Recall (macro): {self.recall_macro:.4f}\n"
            f"\n{self.classification_report}"
        )


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> EvaluationResult:
    """
    Calcula métricas de classificação.

    Args:
        y_true: Rótulos verdadeiros
        y_pred: Rótulos preditos
        labels: Lista ordenada de rótulos (opcional)

    Returns:
        EvaluationResult com todas as métricas
    """
    if not y_true or not y_pred:
        logger.warning("Listas vazias — retornando resultado vazio")
        return EvaluationResult()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Tamanhos incompatíveis: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    result = EvaluationResult(
        accuracy=float(acc),
        f1_macro=float(f1_m),
        f1_weighted=float(f1_w),
        precision_macro=float(prec),
        recall_macro=float(rec),
        confusion_matrix=cm.tolist(),
        classification_report=report,
        labels=labels,
        total_samples=len(y_true),
        correct=int(sum(1 for t, p in zip(y_true, y_pred) if t == p)),
    )

    logger.info("Avaliação concluída — Acc: %.4f | F1(macro): %.4f", acc, f1_m)
    return result


def extract_answer_label(text: str) -> str:
    """
    Extrai o rótulo de resposta (Sim/Não/Talvez) do texto gerado pelo modelo.

    Útil para avaliar PubMedQA onde a resposta é classificação.
    """
    text_lower = text.strip().lower()

    # Verifica início do texto
    for label, keywords in [
        ("sim", ["sim", "yes", "positivo", "correto"]),
        ("não", ["não", "nao", "no", "negativo", "incorreto"]),
        ("talvez", ["talvez", "maybe", "possivelmente", "incerto"]),
    ]:
        for kw in keywords:
            if text_lower.startswith(kw):
                return label

    # Verifica presença em todo o texto
    for label, keywords in [
        ("sim", ["sim", "yes"]),
        ("não", ["não", "nao", "no"]),
        ("talvez", ["talvez", "maybe"]),
    ]:
        for kw in keywords:
            if kw in text_lower:
                return label

    return "desconhecido"


def compute_exact_match(references: list[str], predictions: list[str]) -> float:
    """Calcula Exact Match (EM) — proporção de correspondências exatas."""
    if not references:
        return 0.0
    matches = sum(1 for r, p in zip(references, predictions) if r.strip().lower() == p.strip().lower())
    return matches / len(references)


def compute_token_f1(reference: str, prediction: str) -> float:
    """Calcula F1 em nível de token entre referência e predição."""
    ref_tokens = set(reference.lower().split())
    pred_tokens = set(prediction.lower().split())

    if not ref_tokens or not pred_tokens:
        return 0.0

    common = ref_tokens & pred_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_average_token_f1(
    references: list[str], predictions: list[str]
) -> float:
    """Calcula F1 médio em nível de token."""
    if not references:
        return 0.0
    scores = [compute_token_f1(r, p) for r, p in zip(references, predictions)]
    return float(np.mean(scores))
