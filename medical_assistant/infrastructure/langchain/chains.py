"""
Chains LangChain para o assistente médico.

Define os pipelines de processamento que integram
retriever (RAG) + LLM + prompts para diferentes cenários.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever

from medical_assistant.infrastructure.langchain.memory import create_memory
from medical_assistant.infrastructure.langchain.prompts import (
    MEDICAL_QA_TEMPLATE,
    TRIAGE_TEMPLATE,
    TREATMENT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def create_medical_qa_chain(
    llm: BaseLLM,
    retriever: BaseRetriever,
    verbose: bool = False,
) -> RetrievalQA:
    """
    Cria chain de QA médico com RAG.

    Pipeline: Pergunta → Retriever → Contexto + Prompt → LLM → Resposta com fontes
    """
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=verbose,
        chain_type_kwargs={
            "prompt": MEDICAL_QA_TEMPLATE,
        },
    )
    logger.info("Medical QA Chain criada (stuff + RAG)")
    return chain


def create_conversational_chain(
    llm: BaseLLM,
    retriever: BaseRetriever,
    memory_k: int = 5,
    verbose: bool = False,
) -> ConversationalRetrievalChain:
    """
    Cria chain conversacional com memória e RAG.

    Mantém contexto das últimas K interações para perguntas de follow-up.
    """
    memory = create_memory(k=memory_k)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=verbose,
    )
    logger.info("Conversational Chain criada (memória k=%d)", memory_k)
    return chain


def create_triage_chain(
    llm: BaseLLM,
    verbose: bool = False,
) -> Any:
    """
    Cria chain de triagem (sem RAG, apenas LLM).

    Recebe dados do paciente e classifica urgência.
    """
    from langchain.chains import LLMChain

    chain = LLMChain(
        llm=llm,
        prompt=TRIAGE_TEMPLATE,
        verbose=verbose,
    )
    logger.info("Triage Chain criada")
    return chain


def create_treatment_chain(
    llm: BaseLLM,
    retriever: BaseRetriever | None = None,
    verbose: bool = False,
) -> Any:
    """
    Cria chain de sugestão de tratamento.

    Se retriever disponível, busca protocolos relevantes via RAG.
    """
    if retriever:
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=verbose,
            chain_type_kwargs={
                "prompt": TREATMENT_TEMPLATE,
            },
        )
    else:
        from langchain.chains import LLMChain

        chain = LLMChain(
            llm=llm,
            prompt=TREATMENT_TEMPLATE,
            verbose=verbose,
        )

    logger.info("Treatment Chain criada (RAG=%s)", retriever is not None)
    return chain
