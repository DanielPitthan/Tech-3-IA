"""
Use Case: Avaliar qualidade do modelo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvaluateModel:
    """
    Use case: executa benchmark de avaliação do modelo no test set.

    Métricas:
    - Accuracy e F1 macro/weighted (PubMedQA)
    - LLM-as-judge (qualidade, segurança, relevância)
    """

    def __init__(
        self,
        llm_service: Any,
        test_data_path: str | Path | None = None,
    ):
        self.llm = llm_service
        self.test_data_path = test_data_path

    def execute(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        """Executa avaliação completa."""
        from medical_assistant.evaluation.benchmark import run_benchmark

        results = run_benchmark(
            llm_service=self.llm,
            test_data_path=self.test_data_path,
            output_dir=output_dir,
        )
        return results
