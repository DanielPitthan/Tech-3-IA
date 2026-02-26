"""
Adapter do modelo Falcon-7B para inferência.

Implementa a interface LLMService para integração com LangChain
e o restante da aplicação. Carrega o modelo base + adapter LoRA
para geração de respostas.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.domain.entities.medical_response import MedicalResponse
from medical_assistant.domain.value_objects.confidence_score import ConfidenceScore
from medical_assistant.infrastructure.llm.model_config import ModelConfig

logger = logging.getLogger(__name__)


class FalconModelAdapter(LLMService):
    """
    Adapter para o modelo Falcon-7B fine-tunado.

    Carrega o modelo base com quantização 4-bit e o adapter LoRA
    treinado com dados médicos. Fornece interface unificada para
    geração de texto.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        adapter_path: str | Path | None = None,
    ):
        """
        Args:
            config: Configuração do modelo
            adapter_path: Caminho para o adapter LoRA salvo.
                          Se None, usa apenas o modelo base.
        """
        self.config = config or ModelConfig.from_yaml()
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.model = None
        self.tokenizer = None
        self._pipeline = None
        self._loaded = False

    def load(self) -> None:
        """Carrega modelo e tokenizer para inferência."""
        if self._loaded:
            logger.warning("Modelo já carregado.")
            return

        logger.info("Carregando modelo para inferência: %s", self.config.model_name)

        # Quantização 4-bit
        bnb_config = BitsAndBytesConfig(**self.config.get_bnb_config_dict())

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Modelo base
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
        )

        # Carregar adapter LoRA se disponível
        if self.adapter_path and self.adapter_path.exists():
            logger.info("Carregando adapter LoRA de: %s", self.adapter_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.adapter_path),
            )
            self.model = self.model.merge_and_unload()
            logger.info("Adapter LoRA carregado e mesclado.")
        else:
            logger.info("Usando modelo base sem adapter (fine-tuning não aplicado).")

        # Pipeline de geração
        self._pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        self._loaded = True
        logger.info("Modelo pronto para inferência.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Gera texto a partir de um prompt.

        Args:
            prompt: Texto de entrada
            max_new_tokens: Máximo de tokens a gerar
            temperature: Temperatura de amostragem

        Returns:
            Texto gerado (apenas a parte nova, sem o prompt)
        """
        if not self._loaded:
            self.load()

        gen_config = {
            "max_new_tokens": max_new_tokens or self.config.generation.max_new_tokens,
            "temperature": temperature or self.config.generation.temperature,
            "top_p": self.config.generation.top_p,
            "top_k": self.config.generation.top_k,
            "repetition_penalty": self.config.generation.repetition_penalty,
            "do_sample": self.config.generation.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        gen_config.update(kwargs)

        result = self._pipeline(
            prompt,
            return_full_text=False,
            **gen_config,
        )

        generated_text = result[0]["generated_text"].strip()
        return generated_text

    def generate_medical_response(
        self,
        question: str,
        context: str = "",
        sources: list[str] | None = None,
    ) -> MedicalResponse:
        """
        Gera uma resposta médica estruturada.

        Args:
            question: Pergunta clínica
            context: Contexto adicional (dados do paciente, protocolos)
            sources: Fontes de informação utilizadas

        Returns:
            MedicalResponse com resposta, fontes e confiança
        """
        # Montar prompt no formato de instrução
        prompt = self._format_medical_prompt(question, context)

        # Gerar resposta
        raw_response = self.generate(prompt)

        # Calcular score de confiança (heurística baseada no tamanho e qualidade)
        confidence = self._estimate_confidence(raw_response, context)

        return MedicalResponse(
            response_text=raw_response,
            sources=sources or [],
            confidence=ConfidenceScore(confidence),
            question=question,
            context_used=context,
            model_name=self.config.model_name,
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
        score = 0.5  # Base

        # Bonus por ter contexto
        if context:
            score += 0.15

        # Resposta muito curta = menos confiável
        if len(response) < 50:
            score -= 0.15
        elif len(response) > 200:
            score += 0.1

        # Termos de incerteza reduzem confiança
        uncertainty_terms = ["talvez", "possivelmente", "incerto", "não tenho certeza"]
        for term in uncertainty_terms:
            if term in response.lower():
                score -= 0.05

        return max(0.0, min(1.0, score))

    @property
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado."""
        return self._loaded

    def unload(self) -> None:
        """Libera modelo da memória."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Modelo descarregado da memória.")
