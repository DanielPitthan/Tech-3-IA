"""
Use Case: Processar paciente - executa fluxo clínico completo.
"""

from __future__ import annotations

import logging
from typing import Any

from medical_assistant.application.dtos.patient_dto import (
    ClinicalFlowResultDTO,
    PatientDTO,
    ResponseDTO,
)
from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.domain.entities.alert import Alert, AlertSeverity, AlertType
from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.services.triage_service import (
    ExamService,
    TreatmentService,
    TriageService,
)
from medical_assistant.infrastructure.logging.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ProcessPatient:
    """
    Use case: processar dados de um paciente pelo fluxo clínico completo.

    Etapas:
    1. Triagem
    2. Verificação de exames pendentes
    3. Sugestão de exames adicionais
    4. Sugestão de tratamento (via LLM com RAG)
    5. Checagem de alertas (interações, alergias)
    6. Decisão sobre necessidade de validação humana
    """

    def __init__(
        self,
        llm_service: LLMService,
        retriever: RetrieverService | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.llm = llm_service
        self.retriever = retriever
        self.audit_logger = audit_logger
        self.triage_service = TriageService()
        self.exam_service = ExamService()
        self.treatment_service = TreatmentService()

    def execute(self, patient_dto: PatientDTO) -> ClinicalFlowResultDTO:
        """Executa o fluxo clínico completo para um paciente."""
        logger.info("Processando paciente: %s", patient_dto.id)

        # Converter DTO para entidade
        patient = Patient.from_dict(patient_dto.__dict__)
        flow_log: list[dict[str, Any]] = []

        # 1. Triagem
        triage_level = self.triage_service.classify(patient)
        flow_log.append({"step": "triagem", "result": triage_level.value})
        logger.info("Triagem: %s (%s)", triage_level.value, triage_level.description)

        # 2. Exames pendentes
        pending_exams = self.exam_service.get_pending_exams(patient)
        flow_log.append({"step": "exames_pendentes", "count": len(pending_exams)})

        # 3. Sugestão de exames
        suggested_exams = self.exam_service.suggest_exams(patient)
        flow_log.append({"step": "exames_sugeridos", "exams": suggested_exams})

        # 4. Sugestão de tratamento via LLM
        clinical_summary = patient.to_clinical_summary()
        treatment_response = self._get_treatment_suggestion(clinical_summary)
        flow_log.append({"step": "tratamento", "generated": True})

        # 5. Alertas
        alerts = self._check_alerts(patient)
        flow_log.append({"step": "alertas", "count": len(alerts)})

        # 6. Validação humana
        requires_validation = (
            triage_level.value == "critico"
            or any(a.requires_immediate_action for a in alerts)
            or treatment_response.confidence.value < 0.5
        )
        validation_reason = ""
        if requires_validation:
            reasons = []
            if triage_level.value == "critico":
                reasons.append("paciente em estado crítico")
            if any(a.requires_immediate_action for a in alerts):
                reasons.append("alertas de alta severidade")
            if treatment_response.confidence.value < 0.5:
                reasons.append("baixa confiança na sugestão de tratamento")
            validation_reason = "; ".join(reasons)

        flow_log.append({
            "step": "validacao",
            "required": requires_validation,
            "reason": validation_reason,
        })

        # Audit log
        if self.audit_logger:
            self.audit_logger.log_clinical_flow(
                patient_id=patient_dto.id,
                flow_log=flow_log,
            )

        return ClinicalFlowResultDTO(
            patient_id=patient_dto.id,
            triage_level=triage_level.value,
            triage_description=triage_level.description,
            pending_exams=[e for e in pending_exams],
            suggested_exams=suggested_exams,
            treatment_suggestions=treatment_response.response_text,
            alerts=[a.to_dict() for a in alerts],
            requires_human_validation=requires_validation,
            validation_reason=validation_reason,
            response=ResponseDTO(
                response_text=treatment_response.format_for_display(),
                sources=treatment_response.sources,
                confidence=float(treatment_response.confidence),
                confidence_label=treatment_response.confidence.label,
                disclaimer=treatment_response.disclaimer,
            ),
            flow_log=flow_log,
        )

    def _get_treatment_suggestion(self, clinical_summary: str) -> Any:
        """Gera sugestão de tratamento usando LLM + RAG."""
        context = ""
        sources: list[str] = []

        if self.retriever:
            docs = self.retriever.retrieve(clinical_summary)
            context = "\n\n".join([d.get("content", "") for d in docs])
            sources = [d.get("metadata", {}).get("source_url", "Protocolo interno") for d in docs]

        return self.llm.generate_medical_response(
            question=f"Com base no quadro clínico a seguir, sugira condutas terapêuticas:\n{clinical_summary}",
            context=context,
            sources=sources,
        )

    def _check_alerts(self, patient: Patient) -> list[Alert]:
        """Verifica e gera alertas para o paciente."""
        alerts: list[Alert] = []

        # Sinais vitais críticos
        if patient.has_critical_vitals:
            alerts.append(Alert(
                alert_type=AlertType.SINAL_VITAL_CRITICO,
                severity=AlertSeverity.CRITICA,
                message="Sinais vitais em faixas críticas",
                patient_id=patient.id,
                details=str(patient.sinais_vitais),
                recommendations=["Reavaliação imediata", "Considerar monitorização contínua"],
            ))

        # Verificar alergias vs medicamentos em uso
        for med in patient.medicamentos_em_uso:
            allergy_alerts = self.treatment_service.check_allergies(patient, med["nome"])
            for msg in allergy_alerts:
                alerts.append(Alert(
                    alert_type=AlertType.ALERGIA,
                    severity=AlertSeverity.CRITICA,
                    message=msg,
                    patient_id=patient.id,
                ))

        # Verificar interações medicamentosas entre medicamentos em uso
        meds = [m["nome"] for m in patient.medicamentos_em_uso]
        for i, med_a in enumerate(meds):
            for med_b in meds[i + 1:]:
                interactions = self.treatment_service.check_drug_interactions(
                    patient, med_a
                )
                for msg in interactions:
                    if med_b in msg:
                        alerts.append(Alert(
                            alert_type=AlertType.INTERACAO_MEDICAMENTOSA,
                            severity=AlertSeverity.ALTA,
                            message=msg,
                            patient_id=patient.id,
                        ))

        return alerts
