"""
Value Object: Nível de Triagem.

Classifica a urgência do atendimento do paciente.
"""

from __future__ import annotations

from enum import Enum


class TriageLevel(str, Enum):
    """Nível de triagem clínica (Protocolo de Manchester simplificado)."""

    CRITICO = "critico"
    URGENTE = "urgente"
    REGULAR = "regular"

    @property
    def description(self) -> str:
        descriptions = {
            TriageLevel.CRITICO: "Risco iminente de vida. Atendimento imediato.",
            TriageLevel.URGENTE: "Condição séria. Atendimento prioritário.",
            TriageLevel.REGULAR: "Condição estável. Atendimento conforme ordem de chegada.",
        }
        return descriptions[self]

    @property
    def max_wait_minutes(self) -> int:
        waits = {
            TriageLevel.CRITICO: 0,
            TriageLevel.URGENTE: 30,
            TriageLevel.REGULAR: 120,
        }
        return waits[self]
