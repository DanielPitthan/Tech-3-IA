"""
Nó de Alertas — Identifica riscos e emite alertas médicos.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.domain.entities.alert import Alert, AlertSeverity, AlertType
from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.services.triage_service import TreatmentService
from medical_assistant.infrastructure.langgraph.state import ClinicalState

logger = logging.getLogger(__name__)


def alert_node(state: ClinicalState) -> dict[str, Any]:
    """
    Nó de alertas: identifica riscos e emite alertas médicos.

    Verifica:
    - Sinais vitais críticos
    - Interações medicamentosas
    - Alergias
    - Resultados de exames anormais
    """
    logger.info("🚨 Executando nó de ALERTAS")

    patient_data = state.get("patient_data", {})
    patient = Patient.from_dict(patient_data)
    treatment_service = TreatmentService()

    alerts: list[dict[str, Any]] = []

    # 1. Sinais vitais críticos
    if patient.has_critical_vitals:
        sv = patient.sinais_vitais
        alert = Alert(
            alert_type=AlertType.SINAL_VITAL_CRITICO,
            severity=AlertSeverity.CRITICA,
            message=f"Sinais vitais críticos: PA {sv.get('pa_sistolica', '?')}/{sv.get('pa_diastolica', '?')}, "
                    f"FC {sv.get('fc', '?')}, SpO2 {sv.get('spo2', '?')}%",
            patient_id=patient.id,
            recommendations=[
                "Reavaliação imediata dos sinais vitais",
                "Considerar monitorização contínua",
                "Avaliar necessidade de leito UTI",
            ],
        )
        alerts.append(alert.to_dict())
        logger.warning("ALERTA CRÍTICO: Sinais vitais - Paciente %s", patient.id)

    # 2. Verificar alergias vs medicamentos
    for med in patient.medicamentos_em_uso:
        allergy_alerts = treatment_service.check_allergies(patient, med["nome"])
        for msg in allergy_alerts:
            alert = Alert(
                alert_type=AlertType.ALERGIA,
                severity=AlertSeverity.CRITICA,
                message=msg,
                patient_id=patient.id,
                recommendations=["Suspender medicamento IMEDIATAMENTE", "Avaliar alternativa terapêutica"],
            )
            alerts.append(alert.to_dict())

    # 3. Verificar interações medicamentosas
    meds = [m["nome"] for m in patient.medicamentos_em_uso]
    checked_pairs: set[tuple[str, str]] = set()
    for med in meds:
        interactions = treatment_service.check_drug_interactions(patient, med)
        for msg in interactions:
            # Evitar duplicatas
            pair = tuple(sorted([med, msg]))
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                alert = Alert(
                    alert_type=AlertType.INTERACAO_MEDICAMENTOSA,
                    severity=AlertSeverity.ALTA,
                    message=msg,
                    patient_id=patient.id,
                    recommendations=["Avaliar risco-benefício", "Considerar ajuste de dose ou substituição"],
                )
                alerts.append(alert.to_dict())

    # 4. Resultados de exames críticos
    for exam in patient.available_results:
        resultado = exam.get("resultado", "")
        critical_findings = _check_critical_result(exam["nome"], resultado)
        if critical_findings:
            alert = Alert(
                alert_type=AlertType.EXAME_CRITICO,
                severity=AlertSeverity.ALTA,
                message=f"Resultado crítico em {exam['nome']}: {critical_findings}",
                patient_id=patient.id,
                recommendations=["Reavaliar resultado", "Considerar conduta imediata"],
            )
            alerts.append(alert.to_dict())

    log_entry = {
        "step": "alertas",
        "total_alerts": len(alerts),
        "critical_alerts": sum(1 for a in alerts if a.get("severity") == "critica"),
    }

    logger.info("Alertas gerados: %d (críticos: %d)", len(alerts), log_entry["critical_alerts"])

    return {
        "alerts": alerts,
        "flow_log": [log_entry],
    }


def _check_critical_result(exam_name: str, resultado: str) -> str | None:
    """Verifica se um resultado de exame é crítico."""
    if not resultado:
        return None

    try:
        # Potássio
        if "potássio" in exam_name.lower():
            import re
            match = re.search(r"([\d.]+)", resultado)
            if match:
                val = float(match.group(1))
                if val > 5.5:
                    return f"Hipercalemia ({val} mEq/L)"
                if val < 3.0:
                    return f"Hipocalemia ({val} mEq/L)"

        # SpO2
        if "spo2" in resultado.lower():
            import re
            match = re.search(r"([\d.]+)", resultado)
            if match and float(match.group(1)) < 90:
                return f"Hipoxemia (SpO2 {match.group(1)}%)"

        # INR
        if "inr" in exam_name.lower():
            import re
            match = re.search(r"([\d.]+)", resultado)
            if match:
                val = float(match.group(1))
                if val > 4.0:
                    return f"INR elevado ({val}) — risco de sangramento"

        # Creatinina
        if "creatinina" in exam_name.lower():
            import re
            match = re.search(r"([\d.]+)", resultado)
            if match:
                val = float(match.group(1))
                if val > 3.0:
                    return f"Creatinina elevada ({val} mg/dL) — insuficiência renal"

    except (ValueError, AttributeError):
        pass

    return None
