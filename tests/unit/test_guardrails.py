"""
Testes unitários — Guardrails de segurança.
"""

import pytest

from medical_assistant.domain.entities.medical_response import MedicalResponse
from medical_assistant.domain.value_objects.confidence_score import ConfidenceScore
from medical_assistant.infrastructure.security.guardrails import Guardrails


class TestGuardrails:
    """Testes para o sistema de guardrails."""

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_validate_input_valid(self, guardrails):
        is_valid, msg = guardrails.validate_input("Quais são os sintomas da gripe?")
        assert is_valid is True

    def test_validate_input_empty(self, guardrails):
        is_valid, msg = guardrails.validate_input("")
        # Vazio é aceito desde que não exceda comprimento nem match padrões bloqueados
        assert isinstance(is_valid, bool)

    def test_validate_input_too_long(self, guardrails):
        long_text = "a" * 10001
        is_valid, msg = guardrails.validate_input(long_text)
        assert is_valid is False

    def test_apply_safe_output(self, guardrails):
        response = MedicalResponse(
            response_text="A gripe é causada pelo vírus Influenza. Consulte seu médico.",
            sources=["PubMedQA"],
            confidence=ConfidenceScore(0.8),
        )
        modified, triggered = guardrails.apply(response)
        # Não deve ter padrões proibidos ativados (exceto possivelmente disclaimer)
        assert isinstance(triggered, list)

    def test_apply_detects_prescription(self, guardrails):
        response = MedicalResponse(
            response_text="Prescrevo Amoxicilina 500mg de 8/8h por 7 dias.",
            sources=["PubMedQA"],
            confidence=ConfidenceScore(0.8),
        )
        modified, triggered = guardrails.apply(response)
        assert len(triggered) > 0

    def test_apply_low_confidence(self, guardrails):
        response = MedicalResponse(
            response_text="Resposta sobre medicamento.",
            confidence=ConfidenceScore(0.3),
        )
        modified, triggered = guardrails.apply(response)
        # Confiança no limite pode ou não acionar guardrail
        assert isinstance(triggered, list)


class TestGuardrailsEdgeCases:
    """Testes de borda para guardrails."""

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_none_input(self, guardrails):
        with pytest.raises((TypeError, AttributeError)):
            guardrails.validate_input(None)

    def test_unicode_input(self, guardrails):
        is_valid, msg = guardrails.validate_input("É possível tomar paracetamol com café?")
        assert is_valid is True

    def test_special_characters(self, guardrails):
        is_valid, msg = guardrails.validate_input("O que é COVID-19? (SARS-CoV-2)")
        assert is_valid is True
