"""
Benchmark Runner — Pipeline completo de avaliação (métricas + LLM-as-judge).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.evaluation.metrics import (
    EvaluationResult,
    compute_average_token_f1,
    compute_exact_match,
    compute_metrics,
    extract_answer_label,
)
from medical_assistant.evaluation.llm_judge import JudgeReport, LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Resultado completo de benchmark."""

    model_name: str = ""
    dataset_name: str = ""
    classification_metrics: EvaluationResult | None = None
    exact_match: float = 0.0
    token_f1: float = 0.0
    judge_report: JudgeReport | None = None
    inference_time_seconds: float = 0.0
    samples_evaluated: int = 0
    predictions: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "samples_evaluated": self.samples_evaluated,
            "inference_time_seconds": round(self.inference_time_seconds, 2),
            "avg_inference_time_per_sample": (
                round(self.inference_time_seconds / self.samples_evaluated, 3)
                if self.samples_evaluated > 0
                else 0.0
            ),
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
        }
        if self.classification_metrics:
            result["classification_metrics"] = self.classification_metrics.to_dict()
        if self.judge_report:
            result["llm_judge"] = self.judge_report.to_dict()
        return result

    def summary(self) -> str:
        lines = [
            f"=== Benchmark: {self.model_name} ===",
            f"Dataset: {self.dataset_name}",
            f"Amostras: {self.samples_evaluated}",
            f"Tempo total: {self.inference_time_seconds:.1f}s",
            f"Exact Match: {self.exact_match:.4f}",
            f"Token F1: {self.token_f1:.4f}",
        ]
        if self.classification_metrics:
            lines.append(f"Accuracy: {self.classification_metrics.accuracy:.4f}")
            lines.append(f"F1 (macro): {self.classification_metrics.f1_macro:.4f}")
        if self.judge_report:
            lines.append(f"LLM Judge (nota geral): {self.judge_report.avg_geral:.2f}/5")
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Executa avaliação completa de um modelo em um dataset de teste.

    Pipeline:
    1. Carrega dataset de teste (JSONL)
    2. Gera predições do modelo
    3. Calcula métricas de classificação (PubMedQA: Sim/Não/Talvez)
    4. Calcula Exact Match e Token F1
    5. (Opcional) Executa LLM-as-Judge
    6. Gera relatório consolidado
    """

    def __init__(
        self,
        llm_service: LLMService,
        judge: LLMJudge | None = None,
        output_dir: str = "evaluation_results",
    ) -> None:
        self._llm_service = llm_service
        self._judge = judge
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def load_test_data(self, test_file: str) -> list[dict[str, str]]:
        """
        Carrega dataset de teste (JSONL).

        Espera campos: instruction, input, output
        """
        data: list[dict[str, str]] = []
        path = Path(test_file)

        if not path.exists():
            raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_file}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    data.append(entry)

        logger.info("Carregadas %d amostras de teste de %s", len(data), test_file)
        return data

    def generate_predictions(
        self,
        test_data: list[dict[str, str]],
        max_samples: int | None = None,
    ) -> tuple[list[str], list[str], list[str], float]:
        """
        Gera predições do modelo para os dados de teste.

        Returns:
            (questions, references, predictions, inference_time)
        """
        n = min(len(test_data), max_samples) if max_samples else len(test_data)
        questions: list[str] = []
        references: list[str] = []
        predictions: list[str] = []

        logger.info("Gerando predições para %d amostras", n)
        start_time = time.time()

        for i, sample in enumerate(test_data[:n]):
            instruction = sample.get("instruction", "")
            context = sample.get("input", "")
            reference = sample.get("output", "")

            prompt = instruction
            if context:
                prompt = f"{instruction}\n\nContexto: {context}"

            try:
                prediction = self._llm_service.generate(prompt)
                predictions.append(prediction)
                questions.append(instruction)
                references.append(reference)
            except Exception as e:
                logger.warning("Erro na amostra %d: %s", i, e)
                predictions.append("")
                questions.append(instruction)
                references.append(reference)

            if (i + 1) % 50 == 0:
                logger.info("Progresso: %d/%d predições geradas", i + 1, n)

        elapsed = time.time() - start_time
        logger.info("Predições concluídas em %.1fs", elapsed)

        return questions, references, predictions, elapsed

    def run(
        self,
        test_file: str,
        model_name: str = "llama3-qlora",
        max_samples: int | None = None,
        run_judge: bool = False,
        judge_max_samples: int = 50,
    ) -> BenchmarkResult:
        """
        Executa benchmark completo.

        Args:
            test_file: Caminho para o arquivo de teste JSONL
            model_name: Nome do modelo para referência
            max_samples: Número máximo de amostras para predição
            run_judge: Se deve executar LLM-as-judge
            judge_max_samples: Nº de amostras para o judge

        Returns:
            BenchmarkResult com métricas consolidadas
        """
        logger.info("=== Iniciando Benchmark: %s ===", model_name)

        # 1. Carregar dados
        test_data = self.load_test_data(test_file)

        # 2. Gerar predições
        questions, references, predictions, elapsed = self.generate_predictions(
            test_data, max_samples
        )

        # 3. Métricas de classificação (extrair rótulos)
        true_labels = [extract_answer_label(r) for r in references]
        pred_labels = [extract_answer_label(p) for p in predictions]

        # Filtrar apenas predições com rótulos válidos
        valid_mask = [
            (t != "desconhecido" and p != "desconhecido")
            for t, p in zip(true_labels, pred_labels)
        ]
        filtered_true = [t for t, v in zip(true_labels, valid_mask) if v]
        filtered_pred = [p for p, v in zip(pred_labels, valid_mask) if v]

        classification_result = None
        if filtered_true:
            classification_result = compute_metrics(
                filtered_true,
                filtered_pred,
                labels=["sim", "não", "talvez"],
            )

        # 4. Exact Match e Token F1
        em = compute_exact_match(references, predictions)
        tf1 = compute_average_token_f1(references, predictions)

        # 5. LLM-as-Judge (opcional)
        judge_report = None
        if run_judge and self._judge:
            judge_report = self._judge.judge_batch(
                questions, references, predictions, max_samples=judge_max_samples
            )

        # 6. Consolidar resultado
        result = BenchmarkResult(
            model_name=model_name,
            dataset_name=test_file,
            classification_metrics=classification_result,
            exact_match=em,
            token_f1=tf1,
            judge_report=judge_report,
            inference_time_seconds=elapsed,
            samples_evaluated=len(predictions),
            predictions=[
                {"question": q, "reference": r, "prediction": p}
                for q, r, p in zip(questions, references, predictions)
            ],
        )

        # 7. Salvar resultados
        self._save_results(result, model_name)

        logger.info(result.summary())
        return result

    def run_comparison(
        self,
        test_file: str,
        models: dict[str, LLMService],
        max_samples: int | None = None,
        run_judge: bool = False,
    ) -> dict[str, BenchmarkResult]:
        """
        Executa benchmark comparativo entre múltiplos modelos.

        Args:
            test_file: Arquivo de teste
            models: Mapeamento {nome: serviço LLM}
            max_samples: Limite de amostras
            run_judge: Executar LLM-as-judge

        Returns:
            Mapeamento {nome do modelo: BenchmarkResult}
        """
        results: dict[str, BenchmarkResult] = {}

        for name, service in models.items():
            logger.info("Benchmarking modelo: %s", name)
            original_service = self._llm_service
            self._llm_service = service

            result = self.run(
                test_file=test_file,
                model_name=name,
                max_samples=max_samples,
                run_judge=run_judge,
            )
            results[name] = result
            self._llm_service = original_service

        # Exibir comparativo
        self._print_comparison(results)
        return results

    def _save_results(self, result: BenchmarkResult, model_name: str) -> None:
        """Salva resultados em JSON."""
        filename = f"benchmark_{model_name}_{int(time.time())}.json"
        filepath = self._output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("Resultados salvos em: %s", filepath)

        # Salvar predições separadamente
        pred_file = self._output_dir / f"predictions_{model_name}_{int(time.time())}.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            for pred in result.predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    def _print_comparison(self, results: dict[str, BenchmarkResult]) -> None:
        """Imprime tabela comparativa."""
        header = f"{'Modelo':<25} {'Accuracy':<10} {'F1(macro)':<10} {'EM':<10} {'Token F1':<10} {'Judge':<10}"
        logger.info("=== COMPARATIVO ===")
        logger.info(header)
        logger.info("─" * 75)

        for name, r in results.items():
            acc = r.classification_metrics.accuracy if r.classification_metrics else 0.0
            f1m = r.classification_metrics.f1_macro if r.classification_metrics else 0.0
            judge = r.judge_report.avg_geral if r.judge_report else 0.0
            line = f"{name:<25} {acc:<10.4f} {f1m:<10.4f} {r.exact_match:<10.4f} {r.token_f1:<10.4f} {judge:<10.2f}"
            logger.info(line)
