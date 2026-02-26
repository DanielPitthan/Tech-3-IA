"""
Value Object: Score de Confiança.

Representa a confiança do modelo na resposta gerada.
Valor entre 0.0 e 1.0, com validação.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceScore:
    """Score de confiança da resposta do modelo (0.0 a 1.0)."""

    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                f"ConfidenceScore deve estar entre 0.0 e 1.0, recebido: {self.value}"
            )

    @property
    def label(self) -> str:
        """Rótulo textual do nível de confiança."""
        if self.value >= 0.8:
            return "Alta"
        if self.value >= 0.5:
            return "Média"
        if self.value >= 0.3:
            return "Baixa"
        return "Muito Baixa"

    @property
    def is_reliable(self) -> bool:
        """Indica se a confiança é suficiente para apresentar a resposta."""
        return self.value >= 0.3

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value:.2f} ({self.label})"
