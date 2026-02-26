"""
Validadores de input/output para o assistente médico.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Resultado de uma validação."""

    is_valid: bool
    message: str = ""
    field: str = ""


class InputValidator:
    """Validador de inputs da aplicação."""

    @staticmethod
    def validate_question(question: str) -> ValidationResult:
        """Valida uma pergunta clínica."""
        if not question or not question.strip():
            return ValidationResult(False, "A pergunta não pode estar vazia.", "question")

        if len(question.strip()) < 10:
            return ValidationResult(False, "A pergunta é muito curta (mín. 10 caracteres).", "question")

        if len(question) > 5000:
            return ValidationResult(False, "A pergunta excede o limite de 5000 caracteres.", "question")

        return ValidationResult(True)

    @staticmethod
    def validate_patient_id(patient_id: str) -> ValidationResult:
        """Valida um ID de paciente."""
        if not patient_id or not patient_id.strip():
            return ValidationResult(False, "ID do paciente é obrigatório.", "patient_id")

        return ValidationResult(True)
