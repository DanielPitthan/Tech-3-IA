"""
Nó de Triagem — Classifica a urgência do paciente.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.services.triage_service import TriageService
from medical_assistant.infrastructure.langgraph.state import ClinicalState

logger = logging.getLogger(__name__)


def triage_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de triagem: classifica urgência do paciente.

    Utiliza regras de domínio (sinais vitais, queixa) para
    classificação determinística + LLM para justificativa.
    """
    logger.info("🔍 Executando nó de TRIAGEM")

    patient_data = state.get("patient_data", {})
    patient = Patient.from_dict(patient_data)

    # Classificação por regras de domínio
    triage_service = TriageService()
    triage_level = triage_service.classify(patient)

    # Justificativa
    justification_parts = [f"Classificação: {triage_level.value.upper()}"]
    justification_parts.append(f"Descrição: {triage_level.description}")

    if patient.has_critical_vitals:
        justification_parts.append("⚠️ Sinais vitais em faixas críticas")

    justification_parts.append(f"Queixa principal: {patient.queixa_principal}")
    justification_parts.append(f"Setor: {patient.setor}")

    justification = "\n".join(justification_parts)

    log_entry = {
        "step": "triagem",
        "triage_level": triage_level.value,
        "patient_id": patient.id,
    }

    logger.info("Triagem concluída: %s para paciente %s", triage_level.value, patient.id)

    return {
        "triage_level": triage_level.value,
        "triage_justification": justification,
        "flow_log": [log_entry],
    }
