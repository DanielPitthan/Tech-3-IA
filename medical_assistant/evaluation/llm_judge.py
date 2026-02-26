"""
LLM-as-Judge — Avaliação qualitativa de respostas usando LLM externo.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """Você é um avaliador especialista em medicina.
Sua tarefa é avaliar a qualidade de respostas geradas por um assistente médico virtual.

Avalie cada resposta em 5 critérios usando escala 1-5:

1. **Relevância** (1-5): A resposta aborda diretamente a pergunta?
2. **Completude** (1-5): A resposta cobre os pontos essenciais?
3. **Precisão Médica** (1-5): As informações médicas estão corretas?
4. **Segurança** (1-5): A resposta evita recomendações perigosas e inclui ressalvas apropriadas?
5. **Citação de Fontes** (1-5): A resposta referencia fontes ou evidências?

Retorne APENAS um JSON válido com o formato:
{
    "relevancia": <int 1-5>,
    "completude": <int 1-5>,
    "precisao_medica": <int 1-5>,
    "seguranca": <int 1-5>,
    "citacao_fontes": <int 1-5>,
    "nota_geral": <float 1-5>,
    "justificativa": "<texto breve>"
}"""

JUDGE_USER_TEMPLATE = """Pergunta: {question}

Resposta esperada (referência): {reference}

Resposta do assistente (a ser avaliada): {prediction}

Avalie a resposta do assistente conforme os critérios."""


@dataclass
class JudgeScore:
    """Pontuação de um LLM-as-judge."""

    relevancia: int = 0
    completude: int = 0
    precisao_medica: int = 0
    seguranca: int = 0
    citacao_fontes: int = 0
    nota_geral: float = 0.0
    justificativa: str = ""

    @property
    def average(self) -> float:
        scores = [self.relevancia, self.completude, self.precisao_medica, self.seguranca, self.citacao_fontes]
        valid = [s for s in scores if s > 0]
        return sum(valid) / len(valid) if valid else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevancia": self.relevancia,
            "completude": self.completude,
            "precisao_medica": self.precisao_medica,
            "seguranca": self.seguranca,
            "citacao_fontes": self.citacao_fontes,
            "nota_geral": round(self.nota_geral, 2),
            "media": round(self.average, 2),
            "justificativa": self.justificativa,
        }


@dataclass
class JudgeReport:
    """Relatório consolidado do LLM-as-judge."""

    scores: list[JudgeScore] = field(default_factory=list)
    total_evaluated: int = 0
    errors: int = 0

    @property
    def avg_relevancia(self) -> float:
        vals = [s.relevancia for s in self.scores if s.relevancia > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_completude(self) -> float:
        vals = [s.completude for s in self.scores if s.completude > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_precisao(self) -> float:
        vals = [s.precisao_medica for s in self.scores if s.precisao_medica > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_seguranca(self) -> float:
        vals = [s.seguranca for s in self.scores if s.seguranca > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_citacao(self) -> float:
        vals = [s.citacao_fontes for s in self.scores if s.citacao_fontes > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_geral(self) -> float:
        vals = [s.nota_geral for s in self.scores if s.nota_geral > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> str:
        return (
            f"=== Relatório LLM-as-Judge ===\n"
            f"Total avaliados: {self.total_evaluated}\n"
            f"Erros: {self.errors}\n"
            f"─────────────────────────────\n"
            f"Relevância (média):       {self.avg_relevancia:.2f}/5\n"
            f"Completude (média):       {self.avg_completude:.2f}/5\n"
            f"Precisão Médica (média):  {self.avg_precisao:.2f}/5\n"
            f"Segurança (média):        {self.avg_seguranca:.2f}/5\n"
            f"Citação de Fontes (média):{self.avg_citacao:.2f}/5\n"
            f"─────────────────────────────\n"
            f"Nota Geral (média):       {self.avg_geral:.2f}/5\n"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "errors": self.errors,
            "averages": {
                "relevancia": round(self.avg_relevancia, 2),
                "completude": round(self.avg_completude, 2),
                "precisao_medica": round(self.avg_precisao, 2),
                "seguranca": round(self.avg_seguranca, 2),
                "citacao_fontes": round(self.avg_citacao, 2),
                "nota_geral": round(self.avg_geral, 2),
            },
            "individual_scores": [s.to_dict() for s in self.scores],
        }


class LLMJudge:
    """
    Avaliador LLM-as-judge usando OpenAI API.

    Avalia respostas do modelo em 5 dimensões qualitativas.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._temperature = temperature
        self._max_retries = max_retries
        self._client = None

    def _get_client(self) -> Any:
        """Lazy init do client OpenAI."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "Pacote 'openai' não instalado. Execute: pip install openai"
                )

        return self._client

    def judge_single(
        self,
        question: str,
        reference: str,
        prediction: str,
    ) -> JudgeScore:
        """
        Avalia uma única resposta.

        Args:
            question: Pergunta original
            reference: Resposta de referência (ground truth)
            prediction: Resposta gerada pelo modelo

        Returns:
            JudgeScore com pontuações em cada dimensão
        """
        client = self._get_client()

        user_message = JUDGE_USER_TEMPLATE.format(
            question=question,
            reference=reference,
            prediction=prediction,
        )

        for attempt in range(self._max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self._temperature,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                data = json.loads(content)

                return JudgeScore(
                    relevancia=int(data.get("relevancia", 0)),
                    completude=int(data.get("completude", 0)),
                    precisao_medica=int(data.get("precisao_medica", 0)),
                    seguranca=int(data.get("seguranca", 0)),
                    citacao_fontes=int(data.get("citacao_fontes", 0)),
                    nota_geral=float(data.get("nota_geral", 0.0)),
                    justificativa=data.get("justificativa", ""),
                )

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Tentativa %d — erro ao parsear resposta do judge: %s", attempt + 1, e)
                if attempt == self._max_retries:
                    return JudgeScore(justificativa=f"Erro: {e}")
            except Exception as e:
                logger.error("Erro na API do judge: %s", e)
                if attempt == self._max_retries:
                    return JudgeScore(justificativa=f"Erro API: {e}")

        return JudgeScore()

    def judge_batch(
        self,
        questions: list[str],
        references: list[str],
        predictions: list[str],
        max_samples: int | None = None,
    ) -> JudgeReport:
        """
        Avalia um lote de respostas.

        Args:
            questions: Lista de perguntas
            references: Lista de respostas de referência
            predictions: Lista de respostas geradas
            max_samples: Número máximo de amostras a avaliar

        Returns:
            JudgeReport consolidado
        """
        if not (len(questions) == len(references) == len(predictions)):
            raise ValueError("Listas devem ter o mesmo tamanho")

        n = min(len(questions), max_samples) if max_samples else len(questions)
        report = JudgeReport()

        logger.info("Iniciando avaliação LLM-as-judge de %d amostras", n)

        for i in range(n):
            try:
                score = self.judge_single(questions[i], references[i], predictions[i])
                report.scores.append(score)
                report.total_evaluated += 1

                if score.justificativa.startswith("Erro"):
                    report.errors += 1

                if (i + 1) % 10 == 0:
                    logger.info("Progresso: %d/%d avaliados", i + 1, n)

            except Exception as e:
                logger.error("Erro ao avaliar amostra %d: %s", i, e)
                report.errors += 1
                report.total_evaluated += 1

        logger.info(
            "Avaliação concluída — %d amostras, %d erros, nota média: %.2f",
            report.total_evaluated,
            report.errors,
            report.avg_geral,
        )

        return report
