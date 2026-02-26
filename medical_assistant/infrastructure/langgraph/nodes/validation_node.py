"""
Nó de Validação — Human-in-the-loop para aprovação médica.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.infrastructure.langgraph.state import ClinicalState

logger = logging.getLogger(__name__)


def should_require_validation(state: ClinicalState) -> bool:
    """
    Decide se o fluxo requer validação humana.

    Critérios:
    - Triage = CRITICO ou URGENTE
    - Alertas de severidade CRITICA
    - Confiança < 0.6
    - Interações medicamentosas detectadas
    """
    triage = state.get("triage_level", "regular")
    alerts = state.get("alerts", [])
    confidence = state.get("confidence_score", 1.0)

    if triage in ("critico", "urgente"):
        return True
    if confidence < 0.6:
        return True
    if any(a.get("severity") == "critica" for a in alerts):
        return True
    if any(a.get("alert_type") == "interacao_medicamentosa" for a in alerts):
        return True

    return False


def validation_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de validação: determina se o caso requer revisão humana.

    Não bloqueia o fluxo — apenas marca o estado com o requisito
    de validação. A camada de interface (CLI/API) apresenta ao
    profissional de saúde para aprovação.
    """
    logger.info("✅ Executando nó de VALIDAÇÃO")

    needs_validation = should_require_validation(state)
    reasons: list[str] = []

    triage = state.get("triage_level", "regular")
    if triage in ("critico", "urgente"):
        reasons.append(f"Nível de triagem: {triage}")

    alerts = state.get("alerts", [])
    critical_count = sum(1 for a in alerts if a.get("severity") == "critica")
    if critical_count > 0:
        reasons.append(f"{critical_count} alerta(s) de severidade crítica")

    confidence = state.get("confidence_score", 1.0)
    if confidence < 0.6:
        reasons.append(f"Confiança baixa: {confidence:.2f}")

    interaction_count = sum(1 for a in alerts if a.get("alert_type") == "interacao_medicamentosa")
    if interaction_count > 0:
        reasons.append(f"{interaction_count} interação(ões) medicamentosa(s)")

    validation_reason = "; ".join(reasons) if reasons else "Nenhuma razão — validação não necessária"

    log_entry = {
        "step": "validacao",
        "requires_validation": needs_validation,
        "reasons": reasons,
    }

    logger.info(
        "Validação: %s (%s)",
        "NECESSÁRIA" if needs_validation else "dispensada",
        validation_reason,
    )

    return {
        "human_validation_required": needs_validation,
        "validation_reason": validation_reason,
        "flow_log": [log_entry],
    }


def human_decision_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de decisão humana — ponto de interrupção para o grafo.

    Em produção, o grafo pausa aqui e espera o profissional
    fornecer sua decisão (aprovado|rejeitado|modificado).
    """
    logger.info("🧑‍⚕️ Aguardando decisão humana")

    decision = state.get("human_decision")

    log_entry = {
        "step": "decisao_humana",
        "decision": decision or "pendente",
    }

    if decision is None:
        logger.info("Decisão ainda não fornecida — grafo permanece em pausa")
        return {"flow_log": [log_entry]}

    logger.info("Decisão recebida: %s", decision)

    return {
        "human_decision": decision,
        "flow_log": [log_entry],
    }
