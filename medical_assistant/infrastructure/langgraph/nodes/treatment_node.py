"""
Nó de Sugestão de Tratamento — Gera condutas terapêuticas via LLM + RAG.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.infrastructure.langgraph.state import ClinicalState

logger = logging.getLogger(__name__)

# LLM e retriever são injetados via closure na criação do grafo
_llm_service = None
_retriever_service = None


def set_treatment_dependencies(llm_service: Any, retriever_service: Any = None) -> None:
    """Injeta dependências no nó de tratamento."""
    global _llm_service, _retriever_service
    _llm_service = llm_service
    _retriever_service = retriever_service


def treatment_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de sugestão de tratamento: gera condutas via LLM com RAG.

    Utiliza o contexto do paciente + protocolos recuperados via RAG
    para sugerir condutas terapêuticas.
    """
    logger.info("💊 Executando nó de SUGESTÃO DE TRATAMENTO")

    patient_data = state.get("patient_data", {})
    patient = Patient.from_dict(patient_data)
    clinical_summary = patient.to_clinical_summary()

    # Adicionar análise de exames ao contexto
    exam_analysis = state.get("exam_analysis", "")
    triage_info = state.get("triage_justification", "")

    full_context = f"{clinical_summary}\n\n{exam_analysis}\n\n{triage_info}"

    # Buscar protocolos relevantes via RAG
    sources: list[str] = []
    rag_context = ""
    if _retriever_service:
        try:
            docs = _retriever_service.retrieve(clinical_summary, top_k=5)
            rag_context = "\n\n".join([d.get("content", "") for d in docs])
            sources = [
                d.get("metadata", {}).get("source_url", "Protocolo interno")
                for d in docs
            ]
        except Exception as e:
            logger.warning("Erro no retriever: %s", e)

    # Gerar sugestão via LLM
    suggestion = ""
    confidence = 0.5
    if _llm_service:
        try:
            response = _llm_service.generate_medical_response(
                question=(
                    "Com base no quadro clínico a seguir, sugira condutas terapêuticas "
                    "para avaliação do médico responsável."
                ),
                context=f"{full_context}\n\n{rag_context}",
                sources=sources,
            )
            suggestion = response.response_text
            confidence = float(response.confidence)
        except Exception as e:
            logger.error("Erro na geração de tratamento: %s", e)
            suggestion = (
                "Não foi possível gerar sugestão de tratamento automaticamente. "
                "Consulte os protocolos médicos diretamente."
            )
            confidence = 0.0
    else:
        suggestion = (
            "LLM não disponível. Sugestões de tratamento devem ser consultadas "
            "diretamente nos protocolos médicos do hospital."
        )

    log_entry = {
        "step": "sugestao_tratamento",
        "confidence": confidence,
        "sources_count": len(sources),
        "llm_available": _llm_service is not None,
    }

    return {
        "treatment_suggestion": suggestion,
        "sources": sources,
        "confidence_score": confidence,
        "flow_log": [log_entry],
    }
