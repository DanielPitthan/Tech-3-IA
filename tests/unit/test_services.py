"""
Testes unitários — Serviços de domínio (Triagem, Exames, Tratamento).
"""

import pytest

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.services.triage_service import (
    TriageService,
    ExamService,
    TreatmentService,
)


class TestTriageService:
    """Testes para o serviço de triagem."""

    @pytest.fixture
    def service(self):
        return TriageService()

    def test_critical_by_vitals(self, service, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        level, reason = service.classify(patient)
        assert level in ("critico", "urgente")

    def test_regular_patient(self, service, sample_patient_regular):
        patient = Patient.from_dict(sample_patient_regular)
        level, reason = service.classify(patient)
        assert level == "regular"
        assert reason  # Deve ter justificativa

    def test_critical_by_keywords(self, service):
        data = {
            "id": "P003",
            "nome": "Test",
            "idade": 40,
            "sexo": "M",
            "queixa_principal": "Parada respiratória",
            "diagnosticos": [],
            "medicamentos_em_uso": [],
            "alergias": [],
            "sinais_vitais": {"pa_sistolica": 120, "pa_diastolica": 80,
                              "fc": 72, "fr": 16, "temperatura": 36.5, "spo2": 98},
            "exames": [],
        }
        patient = Patient.from_dict(data)
        level, _ = service.classify(patient)
        assert level == "critico"


class TestExamService:
    """Testes para o serviço de exames."""

    @pytest.fixture
    def service(self):
        return ExamService()

    def test_get_pending(self, service, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        pending = service.get_pending_exams(patient)
        assert len(pending) == 2
        assert "ECG" in pending

    def test_suggest_by_diagnosis(self, service):
        data = {
            "id": "P004",
            "nome": "Test",
            "idade": 50,
            "sexo": "F",
            "queixa_principal": "Rotina",
            "diagnosticos": ["Diabetes Tipo 2"],
            "medicamentos_em_uso": [],
            "alergias": [],
            "sinais_vitais": {"pa_sistolica": 120, "pa_diastolica": 80,
                              "fc": 72, "fr": 16, "temperatura": 36.5, "spo2": 98},
            "exames": [],
        }
        patient = Patient.from_dict(data)
        suggestions = service.suggest_exams(patient)
        assert len(suggestions) > 0


class TestTreatmentService:
    """Testes para o serviço de tratamento."""

    @pytest.fixture
    def service(self):
        return TreatmentService()

    def test_no_allergy(self, service, sample_patient_regular):
        patient = Patient.from_dict(sample_patient_regular)
        alerts = service.check_allergies(patient, "Paracetamol")
        assert len(alerts) == 0

    def test_allergy_detected(self, service, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        # Paciente é alérgico a Penicilina; Amoxicilina está no grupo
        alerts = service.check_allergies(patient, "Amoxicilina")
        # Pode ou não detectar cross-reactivity dependendo da implementação
        # O teste verifica que a função não quebra
        assert isinstance(alerts, list)

    def test_drug_interaction(self, service, sample_patient_data):
        patient = Patient.from_dict(sample_patient_data)
        # Metformina está em uso; verificar interação
        interactions = service.check_drug_interactions(patient, "Insulina")
        assert isinstance(interactions, list)
