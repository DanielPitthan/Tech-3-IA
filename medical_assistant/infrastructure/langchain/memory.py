"""
Memória conversacional para LangChain.

Mantém contexto das interações recentes para perguntas de follow-up.
"""

from __future__ import annotations

from langchain.memory import ConversationBufferWindowMemory


def create_memory(
    k: int = 5,
    memory_key: str = "chat_history",
    return_messages: bool = True,
) -> ConversationBufferWindowMemory:
    """
    Cria memória de buffer com janela deslizante.

    Args:
        k: Número de interações recentes a manter
        memory_key: Chave da memória no chain
        return_messages: Se True, retorna como lista de Messages
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key=memory_key,
        return_messages=return_messages,
        output_key="answer",
    )
