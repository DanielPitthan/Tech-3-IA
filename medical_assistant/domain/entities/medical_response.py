"""
Entidade: Resposta Médica.

Representa a saída do assistente médico com metadados
de explainability (fontes, confiança, disclaimer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from medical_assistant.domain.value_objects.confidence_score import ConfidenceScore


DISCLAIMER = (
    "⚠️ AVISO: Esta é uma sugestão gerada por IA para apoio à decisão clínica. "
    "A decisão final deve ser tomada pelo médico responsável. "
    "Este sistema NÃO substitui o julgamento profissional."
)


@dataclass
class MedicalResponse:
    """Resposta estruturada do assistente médico."""

    response_text: str
    sources: list[str] = field(default_factory=list)
    confidence: ConfidenceScore = field(default_factory=lambda: ConfidenceScore(0.5))
    question: str = ""
    context_used: str = ""
    model_name: str = ""
    disclaimer: str = DISCLAIMER
    guardrails_triggered: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    validated_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_reliable(self) -> bool:
        """Verifica se a resposta tem confiança suficiente."""
        return self.confidence.is_reliable

    @property
    def has_guardrail_warnings(self) -> bool:
        return len(self.guardrails_triggered) > 0

    def format_for_display(self) -> str:
        """Formata resposta para exibição ao usuário."""
        parts = [self.response_text, ""]

        if self.sources:
            parts.append("📚 Fontes:")
            for i, src in enumerate(self.sources, 1):
                parts.append(f"  {i}. {src}")
            parts.append("")

        parts.append(f"🔒 Confiança: {self.confidence}")

        if self.guardrails_triggered:
            parts.append("")
            parts.append("⚠️ Guardrails ativados:")
            for g in self.guardrails_triggered:
                parts.append(f"  - {g}")

        parts.append("")
        parts.append(self.disclaimer)

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serializa para dicionário (logging/auditoria)."""
        return {
            "response_text": self.response_text,
            "sources": self.sources,
            "confidence": float(self.confidence),
            "confidence_label": self.confidence.label,
            "question": self.question,
            "model_name": self.model_name,
            "guardrails_triggered": self.guardrails_triggered,
            "timestamp": self.timestamp.isoformat(),
            "validated_by": self.validated_by,
        }
