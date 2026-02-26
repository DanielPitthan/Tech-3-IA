"""
Nó de Verificação de Exames — Identifica exames pendentes e sugere novos.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.services.triage_service import ExamService
from medical_assistant.infrastructure.langgraph.state import ClinicalState

logger = logging.getLogger(__name__)


def exam_check_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de verificação de exames: identifica pendentes e sugere novos.

    Analisa os exames do paciente e recomenda exames adicionais
    baseados nos diagnósticos.
    """
    logger.info("🔬 Executando nó de VERIFICAÇÃO DE EXAMES")

    patient_data = state.get("patient_data", {})
    patient = Patient.from_dict(patient_data)

    exam_service = ExamService()

    # Exames pendentes
    pending = exam_service.get_pending_exams(patient)
    logger.info("Exames pendentes: %d", len(pending))

    # Resultados disponíveis
    available = exam_service.get_available_results(patient)
    logger.info("Resultados disponíveis: %d", len(available))

    # Sugestão de novos exames
    suggested = exam_service.suggest_exams(patient)
    logger.info("Exames sugeridos: %s", suggested)

    # Análise dos resultados disponíveis
    analysis_parts = []
    if available:
        analysis_parts.append("Resultados de exames disponíveis:")
        for exam in available:
            analysis_parts.append(f"  • {exam['nome']}: {exam.get('resultado', 'N/A')}")
    else:
        analysis_parts.append("Nenhum resultado de exame disponível.")

    if pending:
        analysis_parts.append(f"\nExames pendentes ({len(pending)}):")
        for exam in pending:
            analysis_parts.append(f"  • {exam['nome']} (solicitado em {exam.get('data_solicitacao', 'N/I')})")

    if suggested:
        analysis_parts.append(f"\nExames recomendados adicionais:")
        for exam in suggested:
            analysis_parts.append(f"  • {exam}")

    exam_analysis = "\n".join(analysis_parts)

    log_entry = {
        "step": "verificacao_exames",
        "pending_count": len(pending),
        "available_count": len(available),
        "suggested": suggested,
    }

    return {
        "pending_exams": pending,
        "suggested_exams": suggested,
        "exam_analysis": exam_analysis,
        "flow_log": [log_entry],
    }
