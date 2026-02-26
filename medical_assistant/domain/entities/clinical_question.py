"""
Entidade: Pergunta Clínica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ClinicalQuestion:
    """Pergunta clínica feita por um médico ao assistente."""

    question: str
    patient_context: str = ""
    priority: str = "regular"  # critico | urgente | regular
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_urgent(self) -> bool:
        return self.priority in ("critico", "urgente")

    def to_prompt_context(self) -> str:
        """Formata para uso como contexto do prompt."""
        parts = [f"Pergunta: {self.question}"]
        if self.patient_context:
            parts.insert(0, f"Contexto do paciente:\n{self.patient_context}")
        return "\n\n".join(parts)
