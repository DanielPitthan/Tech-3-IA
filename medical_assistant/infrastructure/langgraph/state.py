"""
State schema para o grafo clínico LangGraph.

Define o estado compartilhado entre todos os nós do fluxo
de decisão clínica.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ClinicalState(TypedDict):
    """Estado do fluxo clínico (compartilhado entre nós do grafo)."""

    # Dados de entrada
    patient_data: dict[str, Any]          # Dados do paciente (JSON)
    question: str                          # Pergunta ou motivo da consulta

    # Resultados intermediários
    triage_level: str                      # critico | urgente | regular
    triage_justification: str              # Justificativa da triagem
    pending_exams: list[dict[str, Any]]    # Exames pendentes
    suggested_exams: list[str]             # Exames sugeridos
    exam_analysis: str                     # Análise dos resultados de exames
    treatment_suggestion: str              # Sugestão de tratamento (LLM)
    alerts: list[dict[str, Any]]           # Alertas gerados
    sources: list[str]                     # Fontes utilizadas

    # Controle de fluxo
    human_validation_required: bool        # Se requer validação humana
    validation_reason: str                 # Motivo da validação
    human_decision: str                    # Decisão humana (aprovado/rejeitado/modificado)

    # Histórico
    messages: Annotated[list, add_messages]  # Histórico de mensagens
    flow_log: list[dict[str, Any]]         # Log de cada etapa do fluxo

    # Metadados
    confidence_score: float
    error: str                             # Erro, se houver
