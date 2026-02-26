"""
Value Object: Status de Exame.
"""

from __future__ import annotations

from enum import Enum


class ExamStatus(str, Enum):
    """Status de um exame médico."""

    PENDENTE = "pendente"
    REALIZADO = "realizado"
    RESULTADO_DISPONIVEL = "resultado_disponivel"
    CANCELADO = "cancelado"

    @property
    def is_actionable(self) -> bool:
        """Retorna True se o exame requer alguma ação."""
        return self in (ExamStatus.PENDENTE,)
