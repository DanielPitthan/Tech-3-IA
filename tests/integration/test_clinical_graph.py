"""
Testes de integração — Nós do LangGraph e fluxo clínico.
"""

import pytest

from medical_assistant.infrastructure.langgraph.nodes.triage_node import triage_node
from medical_assistant.infrastructure.langgraph.nodes.exam_check_node import exam_check_node
from medical_assistant.infrastructure.langgraph.nodes.alert_node import alert_node
from medical_assistant.infrastructure.langgraph.nodes.validation_node import (
    validation_node,
    should_require_validation,
)


class TestTriageNode:
    """Testes para o nó de triagem."""

    def test_critical_patient(self, sample_patient_data):
        state = {"patient_data": sample_patient_data, "question": "Avaliação"}
        result = triage_node(state)
        assert "triage_level" in result
        assert result["triage_level"] in ("critico", "urgente", "regular")
        assert "flow_log" in result

    def test_regular_patient(self, sample_patient_regular):
        state = {"patient_data": sample_patient_regular, "question": "Check-up"}
        result = triage_node(state)
        assert result["triage_level"] == "regular"


class TestExamCheckNode:
    """Testes para o nó de verificação de exames."""

    def test_with_pending_exams(self, sample_patient_data):
        state = {"patient_data": sample_patient_data}
        result = exam_check_node(state)
        assert "pending_exams" in result
        assert len(result["pending_exams"]) == 2

    def test_no_pending_exams(self, sample_patient_regular):
        state = {"patient_data": sample_patient_regular}
        result = exam_check_node(state)
        assert "pending_exams" in result
        assert len(result["pending_exams"]) == 0


class TestAlertNode:
    """Testes para o nó de alertas."""

    def test_critical_vitals_alert(self, sample_patient_data):
        state = {"patient_data": sample_patient_data}
        result = alert_node(state)
        assert "alerts" in result
        assert len(result["alerts"]) > 0
        # Deve ter alerta de sinais vitais críticos
        types = [a.get("alert_type") for a in result["alerts"]]
        assert "sinal_vital_critico" in types

    def test_no_alerts(self, sample_patient_regular):
        state = {"patient_data": sample_patient_regular}
        result = alert_node(state)
        assert "alerts" in result
        # Paciente regular não deve ter alertas
        assert len(result["alerts"]) == 0


class TestValidationNode:
    """Testes para o nó de validação."""

    def test_requires_validation_critical(self):
        state = {
            "triage_level": "critico",
            "alerts": [],
            "confidence_score": 0.8,
        }
        result = validation_node(state)
        assert result["human_validation_required"] is True

    def test_requires_validation_low_confidence(self):
        state = {
            "triage_level": "regular",
            "alerts": [],
            "confidence_score": 0.3,
        }
        result = validation_node(state)
        assert result["human_validation_required"] is True

    def test_no_validation_regular(self):
        state = {
            "triage_level": "regular",
            "alerts": [],
            "confidence_score": 0.9,
        }
        result = validation_node(state)
        assert result["human_validation_required"] is False

    def test_should_require_critical_alert(self):
        state = {
            "triage_level": "regular",
            "alerts": [{"severity": "critica", "alert_type": "sinal_vital_critico"}],
            "confidence_score": 0.8,
        }
        assert should_require_validation(state) is True
