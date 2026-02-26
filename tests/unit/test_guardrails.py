"""
Testes unitários — Guardrails de segurança.
"""

import pytest

from medical_assistant.infrastructure.security.guardrails import Guardrails


class TestGuardrails:
    """Testes para o sistema de guardrails."""

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_validate_input_valid(self, guardrails):
        result = guardrails.validate_input("Quais são os sintomas da gripe?")
        assert result.is_valid is True

    def test_validate_input_empty(self, guardrails):
        result = guardrails.validate_input("")
        assert result.is_valid is False

    def test_validate_input_too_long(self, guardrails):
        long_text = "a" * 10001
        result = guardrails.validate_input(long_text)
        assert result.is_valid is False

    def test_apply_safe_output(self, guardrails):
        output = "A gripe é causada pelo vírus Influenza. Consulte seu médico para avaliação."
        result = guardrails.apply(output, confidence=0.8)
        assert result.is_safe is True

    def test_apply_detects_prescription(self, guardrails):
        output = "Prescrevo Amoxicilina 500mg de 8/8h por 7 dias."
        result = guardrails.apply(output, confidence=0.8)
        # Deve detectar padrão de prescrição
        assert result.is_safe is False or len(result.triggered_rules) > 0

    def test_apply_low_confidence(self, guardrails):
        output = "Resposta sobre medicamento."
        result = guardrails.apply(output, confidence=0.3)
        # Confiança baixa deve acionar guardrail
        assert result.is_safe is False or result.confidence_warning is True


class TestGuardrailsEdgeCases:
    """Testes de borda para guardrails."""

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_none_input(self, guardrails):
        with pytest.raises((TypeError, AttributeError)):
            guardrails.validate_input(None)

    def test_unicode_input(self, guardrails):
        result = guardrails.validate_input("É possível tomar paracetamol com café?")
        assert result.is_valid is True

    def test_special_characters(self, guardrails):
        result = guardrails.validate_input("O que é COVID-19? (SARS-CoV-2)")
        assert result.is_valid is True
