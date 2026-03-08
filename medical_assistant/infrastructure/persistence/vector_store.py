"""
Vector Store — ChromaDB para armazenamento de embeddings.

Indexa documentos médicos (protocolos, respostas MedQuAD) e
fornece busca por similaridade para o pipeline RAG.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Desabilitar telemetry ANTES de importar chromadb (PostHog incompatível)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """
    Vector store baseado em ChromaDB para documentos médicos.

    Armazena embeddings de protocolos, respostas médicas e
    documentação clínica para busca por similaridade (RAG).
    """

    def __init__(
        self,
        persist_directory: str | Path = "./chroma_db",
        collection_name: str = "medical_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._client = None
        self._collection = None
        self._embedding_fn = None

    def _get_embedding_function(self) -> Any:
        """Inicializa função de embedding."""
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions

            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model_name,
            )
        return self._embedding_fn

    def _get_client(self) -> chromadb.ClientAPI:
        """Inicializa cliente ChromaDB com persistência."""
        if self._client is None:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self) -> Any:
        """Obtém ou cria a coleção no ChromaDB."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Adiciona documentos ao vector store.

        Args:
            documents: Lista de dicts com 'content' e 'metadata'
            batch_size: Tamanho do batch para inserção

        Returns:
            Número de documentos adicionados
        """
        collection = self._get_collection()
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]

            ids = []
            texts = []
            metadatas = []

            for j, doc in enumerate(batch):
                doc_id = doc.get("metadata", {}).get("answer_id", f"doc_{i + j}")
                ids.append(str(doc_id))
                texts.append(doc["content"])
                metadatas.append(doc.get("metadata", {}))

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.info(
            "Adicionados %d documentos à coleção '%s'",
            total_added,
            self.collection_name,
        )
        return total_added

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Busca documentos semelhantes à query.

        Args:
            query: Texto de busca
            k: Número de resultados
            where: Filtro de metadata (opcional)

        Returns:
            Lista de documentos com content, metadata e score
        """
        collection = self._get_collection()

        query_params: dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
        }
        if where:
            query_params["where"] = where

        results = collection.query(**query_params)

        documents = []
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                documents.append({
                    "content": doc_text,
                    "metadata": metadata,
                    "score": 1.0 - distance,  # Converter distância em score
                })

        return documents

    @property
    def count(self) -> int:
        """Retorna o número de documentos na coleção."""
        return self._get_collection().count()

    def delete_collection(self) -> None:
        """Remove a coleção inteira."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info("Coleção '%s' removida.", self.collection_name)
        except Exception as e:
            logger.error("Erro ao remover coleção: %s", e)
