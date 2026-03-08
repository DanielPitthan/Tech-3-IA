"""
Testes unitários — Entidades de domínio.
"""

import pytest

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.entities.alert import Alert, AlertSeverity, AlertType
from medical_assistant.domain.entities.medical_response import MedicalResponse
from medical_assistant.domain.entities.clinical_question import ClinicalQuestion
from medical_assistant.domain.value_objects.triage_level import TriageLevel
from medical_assistant.domain.value_objects.exam_status import ExamStatus
from medical_assistant.domain.value_objects.confidence_score import ConfidenceScore


class TestPatient:
    """Testes para a entidade Patient."""

    def test_from_dict(self, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        assert patient.id == "P001"
        assert patient.nome_anonimizado == "João Silva"
        assert patient.idade == 65

    def test_has_critical_vitals_true(self, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        # PA 180/110, FC 105, SpO2 93 → crítico
        assert patient.has_critical_vitals is True

    def test_has_critical_vitals_false(self, sample_patient_regular):
        patient = Patient.from_dict(sample_patient_regular)
        assert patient.has_critical_vitals is False

    def test_pending_exams(self, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        pending = patient.pending_exams
        assert len(pending) == 2
        assert any(e["nome"] == "ECG" for e in pending)

    def test_available_results(self, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        results = patient.available_results
        assert len(results) == 1
        assert results[0]["nome"] == "Hemograma"

    def test_to_clinical_summary(self, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        summary = patient.to_clinical_summary()
        assert "P001" in summary or "João Silva" in summary
        assert "Dor torácica" in summary


class TestAlert:
    """Testes para a entidade Alert."""

    def test_create_alert(self):
        alert = Alert(
            alert_type=AlertType.SINAL_VITAL_CRITICO,
            severity=AlertSeverity.CRITICA,
            message="PA elevada",
            patient_id="P001",
        )
        assert alert.alert_type == AlertType.SINAL_VITAL_CRITICO
        assert alert.severity == AlertSeverity.CRITICA

    def test_format_for_display(self):
        alert = Alert(
            alert_type=AlertType.ALERGIA,
            severity=AlertSeverity.ALTA,
            message="Alergia a Penicilina detectada",
            patient_id="P001",
        )
        display = alert.format_for_display()
        assert "Alergia" in display

    def test_to_dict(self):
        alert = Alert(
            alert_type=AlertType.INTERACAO_MEDICAMENTOSA,
            severity=AlertSeverity.MEDIA,
            message="Interação leve",
            patient_id="P001",
            recommendations=["Monitorar"],
        )
        d = alert.to_dict()
        assert d["alert_type"] == "interacao_medicamentosa"
        assert d["severity"] == "media"
        assert len(d["recommendations"]) == 1


class TestMedicalResponse:
    """Testes para a entidade MedicalResponse."""

    def test_create(self):
        resp = MedicalResponse(
            response_text="Resposta teste",
            sources=["PubMedQA"],
            confidence=ConfidenceScore(0.85),
        )
        assert resp.response_text == "Resposta teste"
        assert resp.confidence.value == 0.85

    def test_format_for_display(self):
        resp = MedicalResponse(
            response_text="Resposta",
            disclaimer="Protótipo acadêmico",
        )
        display = resp.format_for_display()
        assert "Resposta" in display


class TestTriageLevel:
    """Testes para o Value Object TriageLevel."""

    def test_critico(self):
        tl = TriageLevel.CRITICO
        assert tl.value == "critico"
        assert tl.max_wait_minutes == 0

    def test_urgente(self):
        tl = TriageLevel.URGENTE
        assert tl.max_wait_minutes == 30

    def test_regular(self):
        tl = TriageLevel.REGULAR
        assert tl.max_wait_minutes == 120


class TestConfidenceScore:
    """Testes para o Value Object ConfidenceScore."""

    def test_high_confidence(self):
        cs = ConfidenceScore(0.9)
        assert cs.is_reliable is True
        assert cs.label == "Alta"

    def test_low_confidence(self):
        cs = ConfidenceScore(0.3)
        assert cs.is_reliable is True
        assert cs.label == "Baixa"

    def test_boundary(self):
        cs = ConfidenceScore(0.6)
        assert cs.label == "Média"

    def test_clamping(self):
        with pytest.raises(ValueError):
            ConfidenceScore(1.5)
        with pytest.raises(ValueError):
            ConfidenceScore(-0.5)


class TestExamStatus:
    """Testes para o Value Object ExamStatus."""

    def test_values(self):
        assert ExamStatus.PENDENTE.value == "pendente"
        assert ExamStatus.REALIZADO.value == "realizado"
        assert ExamStatus.RESULTADO_DISPONIVEL.value == "resultado_disponivel"
        assert ExamStatus.CANCELADO.value == "cancelado"
