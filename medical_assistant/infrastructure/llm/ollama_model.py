"""
Adapter do modelo Llama 3 via Ollama para inferência.

Implementa a interface LLMService usando o Ollama como backend,
permitindo inferência local sem necessidade de GPU dedicada
ou carregamento manual de pesos via HuggingFace.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import OllamaLLM

from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.domain.entities.medical_response import MedicalResponse
from medical_assistant.domain.value_objects.confidence_score import ConfidenceScore
from medical_assistant.infrastructure.llm.model_config import ModelConfig

logger = logging.getLogger(__name__)


class OllamaModelAdapter(LLMService):
    """
    Adapter para modelos servidos pelo Ollama (ex.: Llama 3).

    Conecta-se ao servidor Ollama local/remoto e utiliza
    langchain-ollama para integração com as chains LangChain.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
    ):
        """
        Args:
            config: Configuração do modelo (carregada do YAML se None).
            model_name: Nome do modelo no Ollama (ex.: 'llama3').
                        Sobrescreve config.model_name se informado.
            base_url: URL do servidor Ollama (ex.: 'http://localhost:11434').
                      Sobrescreve config.base_url se informado.
        """
        self.config = config or ModelConfig.from_yaml()
        self._model_name = model_name or self.config.model_name
        self._base_url = base_url or self.config.base_url
        self._llm: OllamaLLM | None = None
        self._loaded = False

    def load(self) -> None:
        """Inicializa o client Ollama e valida conexão."""
        if self._loaded:
            logger.warning("Modelo Ollama já carregado.")
            return

        logger.info(
            "Conectando ao Ollama — modelo: %s, url: %s",
            self._model_name,
            self._base_url,
        )

        self._llm = OllamaLLM(
            model=self._model_name,
            base_url=self._base_url,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            repeat_penalty=self.config.generation.repetition_penalty,
            num_predict=self.config.generation.max_new_tokens,
        )

        # Validar conexão com uma chamada leve
        try:
            self._llm.invoke("Olá")
            logger.info("Conexão com Ollama validada com sucesso.")
        except Exception as e:
            logger.error(
                "Falha ao conectar com Ollama em %s: %s. "
                "Verifique se o servidor Ollama está em execução e o modelo '%s' "
                "foi baixado (ollama pull %s).",
                self._base_url,
                e,
                self._model_name,
                self._model_name,
            )
            raise ConnectionError(
                f"Não foi possível conectar ao Ollama em {self._base_url}. "
                f"Execute 'ollama serve' e 'ollama pull {self._model_name}'."
            ) from e

        self._loaded = True
        logger.info("Modelo Ollama pronto para inferência: %s", self._model_name)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Gera texto a partir de um prompt via Ollama.

        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a gerar
            temperature: Temperatura de amostragem

        Returns:
            Texto gerado
        """
        if not self._loaded:
            self.load()

        # Criar instância com overrides se necessário
        invoke_kwargs: dict[str, Any] = {}
        if max_new_tokens or temperature:
            llm = OllamaLLM(
                model=self._model_name,
                base_url=self._base_url,
                temperature=temperature or self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                repeat_penalty=self.config.generation.repetition_penalty,
                num_predict=max_new_tokens or self.config.generation.max_new_tokens,
            )
        else:
            llm = self._llm

        result = llm.invoke(prompt, **invoke_kwargs)
        return result.strip()

    def generate_medical_response(
        self,
        question: str,
        context: str = "",
        sources: list[str] | None = None,
    ) -> MedicalResponse:
        """
        Gera uma resposta médica estruturada via Ollama.

        Args:
            question: Pergunta clínica
            context: Contexto adicional (dados do paciente, protocolos)
            sources: Fontes de informação utilizadas

        Returns:
            MedicalResponse com resposta, fontes e confiança
        """
        prompt = self._format_medical_prompt(question, context)
        raw_response = self.generate(prompt)
        confidence = self._estimate_confidence(raw_response, context)

        return MedicalResponse(
            response_text=raw_response,
            sources=sources or [],
            confidence=ConfidenceScore(confidence),
            question=question,
            context_used=context,
            model_name=f"ollama/{self._model_name}",
        )

    def _format_medical_prompt(self, question: str, context: str = "") -> str:
        """Formata prompt no padrão de instrução médica."""
        instruction = (
            "Você é um assistente médico especializado. Responda à pergunta clínica "
            "de forma precisa e baseada em evidências. Sempre indique que a decisão "
            "final deve ser do médico responsável."
        )

        parts = [f"### Instrução:\n{instruction}"]

        if context:
            parts.append(f"### Contexto:\n{context}")

        parts.append(f"### Pergunta:\n{question}")
        parts.append("### Resposta:")

        return "\n\n".join(parts)

    def _estimate_confidence(self, response: str, context: str) -> float:
        """
        Estima score de confiança da resposta (heurística).

        Fatores considerados:
        - Presença de contexto (RAG)
        - Tamanho da resposta
        - Presença de termos de incerteza
        """
        score = 0.5

        if context:
            score += 0.15

        if len(response) < 50:
            score -= 0.15
        elif len(response) > 200:
            score += 0.1

        uncertainty_terms = ["talvez", "possivelmente", "incerto", "não tenho certeza"]
        for term in uncertainty_terms:
            if term in response.lower():
                score -= 0.05

        return max(0.0, min(1.0, score))

    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._loaded

    @property
    def langchain_llm(self) -> OllamaLLM:
        """Retorna a instância OllamaLLM para uso direto com LangChain chains."""
        if not self._loaded:
            self.load()
        return self._llm

    def unload(self) -> None:
        """Libera referências ao client Ollama."""
        self._llm = None
        self._loaded = False
        logger.info("Modelo Ollama descarregado.")
