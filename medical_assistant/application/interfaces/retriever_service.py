"""
Interface abstrata para serviço de retrieval (RAG).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrieverService(ABC):
    """Interface para serviço de busca por similaridade."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Busca documentos relevantes para a query."""
        ...

    @abstractmethod
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Adiciona documentos à base de conhecimento."""
        ...
