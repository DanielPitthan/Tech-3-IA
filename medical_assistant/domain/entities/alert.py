"""
Entidade: Alerta Médico.

Representa alertas gerados pelo sistema para a equipe médica
(interações medicamentosas, alergias, contraindicações, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AlertType(str, Enum):
    """Tipo de alerta médico."""

    INTERACAO_MEDICAMENTOSA = "interacao_medicamentosa"
    ALERGIA = "alergia"
    CONTRAINDICACAO = "contraindicacao"
    EXAME_CRITICO = "exame_critico"
    SINAL_VITAL_CRITICO = "sinal_vital_critico"
    PRIORIDADE_ALTERADA = "prioridade_alterada"
    GERAL = "geral"


class AlertSeverity(str, Enum):
    """Severidade do alerta."""

    CRITICA = "critica"
    ALTA = "alta"
    MEDIA = "media"
    BAIXA = "baixa"


@dataclass
class Alert:
    """Alerta médico emitido pelo sistema."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    patient_id: str = ""
    details: str = ""
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_immediate_action(self) -> bool:
        return self.severity in (AlertSeverity.CRITICA, AlertSeverity.ALTA)

    def format_for_display(self) -> str:
        """Formata alerta para exibição."""
        severity_icons = {
            AlertSeverity.CRITICA: "🔴",
            AlertSeverity.ALTA: "🟠",
            AlertSeverity.MEDIA: "🟡",
            AlertSeverity.BAIXA: "🟢",
        }
        icon = severity_icons.get(self.severity, "⚪")

        parts = [
            f"{icon} ALERTA [{self.severity.value.upper()}]: {self.message}",
        ]

        if self.details:
            parts.append(f"   Detalhes: {self.details}")

        if self.recommendations:
            parts.append("   Recomendações:")
            for rec in self.recommendations:
                parts.append(f"     • {rec}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "patient_id": self.patient_id,
            "details": self.details,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }
