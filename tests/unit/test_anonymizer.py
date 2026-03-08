"""
Testes unitários — Anonymizer (anonimização de dados).
"""

import pytest

from medical_assistant.infrastructure.security.anonymizer import Anonymizer


class TestAnonymizer:
    """Testes para anonimização de dados sensíveis."""

    @pytest.fixture
    def anonymizer(self):
        return Anonymizer()

    def test_cpf_removal(self, anonymizer):
        text = "Paciente CPF 123.456.789-00 apresentou febre."
        result = anonymizer.anonymize(text)
        assert "123.456.789-00" not in result
        assert "[CPF_ANONIMIZADO]" in result

    def test_phone_removal(self, anonymizer):
        text = "Contato: (11) 99876-5432"
        result = anonymizer.anonymize(text)
        assert "99876-5432" not in result

    def test_email_removal(self, anonymizer):
        text = "Email: paciente@hospital.com.br para contato."
        result = anonymizer.anonymize(text)
        assert "paciente@hospital.com.br" not in result

    def test_cep_removal(self, anonymizer):
        text = "Endereço: CEP 01310-100"
        result = anonymizer.anonymize(text)
        assert "01310-100" not in result

    def test_no_sensitive_data(self, anonymizer):
        text = "Paciente apresentou queixa de cefaleia intensa."
        result = anonymizer.anonymize(text)
        assert result == text

    def test_multiple_patterns(self, anonymizer):
        text = "CPF 123.456.789-00, tel (11) 91234-5678, email j@x.com"
        result = anonymizer.anonymize(text)
        assert "123.456.789-00" not in result
        assert "91234-5678" not in result
        assert "j@x.com" not in result
