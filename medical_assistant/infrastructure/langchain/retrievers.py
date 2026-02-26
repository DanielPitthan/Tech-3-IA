"""
Retriever médico baseado em ChromaDB para LangChain.

Encapsula a busca por similaridade no vector store
como um LangChain BaseRetriever.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore

logger = logging.getLogger(__name__)


class MedicalRetriever(BaseRetriever, RetrieverService):
    """
    Retriever médico que busca documentos relevantes no ChromaDB.

    Integra com LangChain como BaseRetriever e implementa a
    interface RetrieverService da camada de aplicação.
    """

    vector_store: MedicalVectorStore
    search_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Implementa busca do BaseRetriever (LangChain)."""
        results = self.vector_store.similarity_search(query, k=self.search_k)

        documents = []
        for r in results:
            doc = Document(
                page_content=r["content"],
                metadata={
                    **r.get("metadata", {}),
                    "relevance_score": r.get("score", 0.0),
                },
            )
            documents.append(doc)

        logger.debug("Retriever retornou %d documentos para query: %s...", len(documents), query[:50])
        return documents

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Implementa interface RetrieverService."""
        return self.vector_store.similarity_search(query, k=top_k)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Implementa interface RetrieverService."""
        self.vector_store.add_documents(documents)
