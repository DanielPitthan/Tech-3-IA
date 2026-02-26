"""
DTOs (Data Transfer Objects) para a camada de aplicação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PatientDTO:
    """DTO para dados de paciente."""

    id: str
    nome_anonimizado: str = ""
    idade: int = 0
    sexo: str = ""
    setor: str = ""
    diagnosticos: list[str] = field(default_factory=list)
    medicamentos_em_uso: list[dict[str, str]] = field(default_factory=list)
    exames: list[dict[str, Any]] = field(default_factory=list)
    alergias: list[str] = field(default_factory=list)
    sinais_vitais: dict[str, float] = field(default_factory=dict)
    queixa_principal: str = ""


@dataclass
class QuestionDTO:
    """DTO para pergunta clínica."""

    question: str
    patient_id: str | None = None
    patient_context: str = ""
    priority: str = "regular"


@dataclass
class ResponseDTO:
    """DTO para resposta do assistente."""

    response_text: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    confidence_label: str = "Média"
    disclaimer: str = ""
    guardrails_triggered: list[str] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    model_name: str = ""


@dataclass
class ClinicalFlowResultDTO:
    """DTO para resultado do fluxo clínico completo."""

    patient_id: str
    triage_level: str = ""
    triage_description: str = ""
    pending_exams: list[dict[str, Any]] = field(default_factory=list)
    suggested_exams: list[str] = field(default_factory=list)
    treatment_suggestions: str = ""
    alerts: list[dict[str, Any]] = field(default_factory=list)
    requires_human_validation: bool = False
    validation_reason: str = ""
    response: ResponseDTO | None = None
    flow_log: list[dict[str, Any]] = field(default_factory=list)
