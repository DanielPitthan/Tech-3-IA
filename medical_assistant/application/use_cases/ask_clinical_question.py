"""
Use Case: Perguntar ao assistente médico.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.application.dtos.patient_dto import QuestionDTO, ResponseDTO
from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.infrastructure.logging.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class AskClinicalQuestion:
    """
    Use case: médico faz uma pergunta clínica ao assistente.

    Pipeline:
    1. Recuperar contexto relevante via RAG (se retriever disponível)
    2. Gerar resposta com a LLM
    3. Aplicar guardrails de segurança
    4. Registrar no audit log
    5. Retornar resposta estruturada
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever: RetrieverService | None = None,
        guardrails: Any = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.llm = llm_service
        self.retriever = retriever
        self.guardrails = guardrails
        self.audit_logger = audit_logger

    def execute(self, question_dto: QuestionDTO) -> ResponseDTO:
        """Executa o use case de pergunta clínica."""
        logger.info("Processando pergunta: %s", question_dto.question[:100])

        # 1. Recuperar contexto via RAG
        context = ""
        sources: list[str] = []
        if self.retriever:
            docs = self.retriever.retrieve(question_dto.question)
            context = "\n\n".join([d.get("content", "") for d in docs])
            sources = [d.get("metadata", {}).get("source_url", "Base de conhecimento") for d in docs]
            logger.info("Contexto RAG recuperado: %d documentos", len(docs))

        # Adicionar contexto do paciente se disponível
        if question_dto.patient_context:
            context = f"Dados do paciente:\n{question_dto.patient_context}\n\n{context}"

        # 2. Gerar resposta com LLM
        medical_response = self.llm.generate_medical_response(
            question=question_dto.question,
            context=context,
            sources=sources,
        )

        # 3. Aplicar guardrails
        guardrails_triggered: list[str] = []
        if self.guardrails:
            medical_response, guardrails_triggered = self.guardrails.apply(medical_response)

        # 4. Montar DTO de resposta
        response_dto = ResponseDTO(
            response_text=medical_response.response_text,
            sources=medical_response.sources,
            confidence=float(medical_response.confidence),
            confidence_label=medical_response.confidence.label,
            disclaimer=medical_response.disclaimer,
            guardrails_triggered=guardrails_triggered,
            model_name=medical_response.model_name,
        )

        # 5. Audit log
        if self.audit_logger:
            self.audit_logger.log_interaction(
                question=question_dto.question,
                response=medical_response.response_text,
                sources=sources,
                confidence=float(medical_response.confidence),
                guardrails=guardrails_triggered,
            )

        return response_dto
