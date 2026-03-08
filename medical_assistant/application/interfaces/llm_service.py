"""
Interface abstrata para serviço de LLM.

Define o contrato que qualquer implementação de modelo
(Llama 3, Ollama, API externa) deve seguir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medical_assistant.domain.entities.medical_response import MedicalResponse


class LLMService(ABC):
    """Interface abstrata para serviço de LLM."""

    @abstractmethod
    def load(self) -> None:
        """Carrega o modelo para inferência."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Gera texto a partir de um prompt."""
        ...

    @abstractmethod
    def generate_medical_response(
        self,
        question: str,
        context: str = "",
        sources: list[str] | None = None,
    ) -> MedicalResponse:
        """Gera resposta médica estruturada."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        ...
