"""
Interfaces abstratas de repositórios (Domain layer).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from medical_assistant.domain.entities.patient import Patient


class PatientRepository(ABC):
    """Interface para repositório de pacientes."""

    @abstractmethod
    def get_by_id(self, patient_id: str) -> Patient | None:
        ...

    @abstractmethod
    def list_all(self) -> list[Patient]:
        ...

    @abstractmethod
    def save(self, patient: Patient) -> None:
        ...


class ProtocolRepository(ABC):
    """Interface para repositório de protocolos médicos."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_by_id(self, protocol_id: str) -> dict[str, Any] | None:
        ...


class KnowledgeRepository(ABC):
    """Interface para repositório de base de conhecimento (vector store)."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        ...
