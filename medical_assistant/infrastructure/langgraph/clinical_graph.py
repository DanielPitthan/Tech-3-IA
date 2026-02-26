"""
Grafo Clínico — Orquestração completa do fluxo com LangGraph.

START → triage → exam_check → treatment → alert → validation → END
                                                      ↓
                                               (human_decision) → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.infrastructure.langgraph.state import ClinicalState
from medical_assistant.infrastructure.langgraph.nodes.triage_node import triage_node
from medical_assistant.infrastructure.langgraph.nodes.exam_check_node import exam_check_node
from medical_assistant.infrastructure.langgraph.nodes.treatment_node import (
    set_treatment_dependencies,
    treatment_node,
)
from medical_assistant.infrastructure.langgraph.nodes.alert_node import alert_node
from medical_assistant.infrastructure.langgraph.nodes.validation_node import (
    validation_node,
    human_decision_node,
    should_require_validation,
)

logger = logging.getLogger(__name__)


def _route_after_validation(state: ClinicalState) -> str:
    """
    Roteamento condicional pós-validação.
    Se validação humana é necessária, envia para nó de decisão;
    caso contrário, finaliza o grafo.
    """
    if state.get("human_validation_required", False):
        return "human_decision"
    return END


class ClinicalGraph:
    """
    Orquestra o fluxo clínico completo usando LangGraph StateGraph.

    Fluxo:
        START → triage → exam_check → treatment → alert → validation
                                                              ↓
                                                    (human_decision) → END

    Uso:
        graph = ClinicalGraph(llm_service, retriever_service)
        result = graph.run(patient_data, question)
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever_service: RetrieverService,
        checkpointer: Any | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._retriever_service = retriever_service
        self._checkpointer = checkpointer or MemorySaver()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Constrói e compila o StateGraph."""

        # Injetar dependências no nó de tratamento
        set_treatment_dependencies(self._llm_service, self._retriever_service)

        # Criar grafo
        builder = StateGraph(ClinicalState)

        # Adicionar nós
        builder.add_node("triage", triage_node)
        builder.add_node("exam_check", exam_check_node)
        builder.add_node("treatment", treatment_node)
        builder.add_node("alert", alert_node)
        builder.add_node("validation", validation_node)
        builder.add_node("human_decision", human_decision_node)

        # Definir arestas
        builder.add_edge(START, "triage")
        builder.add_edge("triage", "exam_check")
        builder.add_edge("exam_check", "treatment")
        builder.add_edge("treatment", "alert")
        builder.add_edge("alert", "validation")

        # Roteamento condicional após validação
        builder.add_conditional_edges(
            "validation",
            _route_after_validation,
            {
                "human_decision": "human_decision",
                END: END,
            },
        )
        builder.add_edge("human_decision", END)

        # Compilar com checkpointer para persistência de estado
        compiled = builder.compile(checkpointer=self._checkpointer)

        logger.info("Grafo clínico compilado com sucesso")
        return compiled

    def run(
        self,
        patient_data: dict[str, Any],
        question: str,
        thread_id: str = "default",
    ) -> ClinicalState:
        """
        Executa o fluxo clínico completo para um paciente.

        Args:
            patient_data: Dados do paciente (dict compatível com Patient.from_dict)
            question: Pergunta clínica ou motivo da consulta
            thread_id: ID do thread para persistência de estado

        Returns:
            ClinicalState com todos os resultados do fluxo
        """
        logger.info(
            "Iniciando fluxo clínico para paciente %s",
            patient_data.get("id", "desconhecido"),
        )

        initial_state: ClinicalState = {
            "patient_data": patient_data,
            "question": question,
            "triage_level": "",
            "triage_justification": "",
            "pending_exams": [],
            "suggested_exams": [],
            "exam_analysis": "",
            "treatment_suggestion": "",
            "alerts": [],
            "sources": [],
            "human_validation_required": False,
            "validation_reason": "",
            "human_decision": None,
            "messages": [],
            "flow_log": [],
            "confidence_score": 0.0,
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = self._graph.invoke(initial_state, config=config)
            logger.info(
                "Fluxo clínico concluído — Triagem: %s | Alertas: %d | Validação: %s",
                result.get("triage_level", "?"),
                len(result.get("alerts", [])),
                "SIM" if result.get("human_validation_required") else "NÃO",
            )
            return result

        except Exception as e:
            logger.error("Erro no fluxo clínico: %s", str(e), exc_info=True)
            initial_state["error"] = str(e)
            return initial_state

    async def arun(
        self,
        patient_data: dict[str, Any],
        question: str,
        thread_id: str = "default",
    ) -> ClinicalState:
        """Versão assíncrona de run()."""
        logger.info(
            "Iniciando fluxo clínico (async) para paciente %s",
            patient_data.get("id", "desconhecido"),
        )

        initial_state: ClinicalState = {
            "patient_data": patient_data,
            "question": question,
            "triage_level": "",
            "triage_justification": "",
            "pending_exams": [],
            "suggested_exams": [],
            "exam_analysis": "",
            "treatment_suggestion": "",
            "alerts": [],
            "sources": [],
            "human_validation_required": False,
            "validation_reason": "",
            "human_decision": None,
            "messages": [],
            "flow_log": [],
            "confidence_score": 0.0,
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = await self._graph.ainvoke(initial_state, config=config)
            return result
        except Exception as e:
            logger.error("Erro no fluxo clínico (async): %s", str(e), exc_info=True)
            initial_state["error"] = str(e)
            return initial_state

    def resume_with_decision(
        self,
        thread_id: str,
        decision: str,
    ) -> ClinicalState:
        """
        Retoma o fluxo após decisão humana.

        Args:
            thread_id: ID do thread em pausa
            decision: "aprovado", "rejeitado", ou "modificado"

        Returns:
            Estado final do fluxo
        """
        logger.info("Retomando fluxo com decisão: %s (thread=%s)", decision, thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        update = {"human_decision": decision}

        try:
            result = self._graph.invoke(update, config=config)
            return result
        except Exception as e:
            logger.error("Erro ao retomar fluxo: %s", str(e), exc_info=True)
            return {"error": str(e)}

    def get_graph_visualization(self) -> str:
        """Retorna representação Mermaid do grafo para documentação."""
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            # Fallback manual
            return """
graph TD
    START([START]) --> triage[Triagem]
    triage --> exam_check[Verificação de Exames]
    exam_check --> treatment[Tratamento]
    treatment --> alert[Alertas]
    alert --> validation[Validação]
    validation -->|Requer validação| human_decision[Decisão Humana]
    validation -->|Validação dispensada| END([END])
    human_decision --> END
"""

    @property
    def graph(self) -> Any:
        """Acesso direto ao grafo compilado."""
        return self._graph


def create_clinical_graph(
    llm_service: LLMService,
    retriever_service: RetrieverService,
) -> ClinicalGraph:
    """Factory function para criar o grafo clínico."""
    return ClinicalGraph(llm_service, retriever_service)
