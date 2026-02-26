"""
Audit Logger — Logging detalhado para rastreamento e auditoria.

Registra todas as interações do assistente médico em formato
JSON estruturado para compliance e análise posterior.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Logger de auditoria para o assistente médico.

    Registra todas as interações, decisões e alertas em formato
    JSON estruturado para compliance regulatório e análise.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        log_file: str = "audit.jsonl",
    ):
        self.log_dir = Path(log_dir or os.environ.get("LOG_DIR", "./logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_file

        # Configurar logger Python
        self._logger = logging.getLogger("medical_assistant.audit")
        if not self._logger.handlers:
            handler = logging.FileHandler(
                self.log_dir / "application.log",
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Escreve uma entrada no arquivo de auditoria JSONL."""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def log_interaction(
        self,
        question: str,
        response: str,
        sources: list[str] | None = None,
        confidence: float = 0.0,
        guardrails: list[str] | None = None,
        user_id: str = "system",
        model_name: str = "",
    ) -> None:
        """Registra uma interação pergunta-resposta."""
        entry = {
            "event_type": "interaction",
            "user_id": user_id,
            "question": question,
            "response": response[:500],  # Limitar tamanho
            "sources": sources or [],
            "confidence": confidence,
            "guardrails_triggered": guardrails or [],
            "model_name": model_name,
        }
        self._write_entry(entry)
        self._logger.info(
            "Interação registrada | Confiança: %.2f | Guardrails: %d",
            confidence,
            len(guardrails or []),
        )

    def log_clinical_flow(
        self,
        patient_id: str,
        flow_log: list[dict[str, Any]],
    ) -> None:
        """Registra execução de fluxo clínico."""
        entry = {
            "event_type": "clinical_flow",
            "patient_id": patient_id,
            "flow_steps": flow_log,
        }
        self._write_entry(entry)
        self._logger.info("Fluxo clínico registrado | Paciente: %s", patient_id)

    def log_alert(
        self,
        patient_id: str,
        alert_type: str,
        severity: str,
        message: str,
    ) -> None:
        """Registra alerta médico."""
        entry = {
            "event_type": "alert",
            "patient_id": patient_id,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
        }
        self._write_entry(entry)
        self._logger.warning(
            "ALERTA [%s] Paciente %s: %s", severity, patient_id, message
        )

    def log_guardrail_trigger(
        self,
        rule_name: str,
        input_text: str,
        action_taken: str,
    ) -> None:
        """Registra ativação de guardrail."""
        entry = {
            "event_type": "guardrail_triggered",
            "rule": rule_name,
            "input_snippet": input_text[:200],
            "action": action_taken,
        }
        self._write_entry(entry)
        self._logger.warning("Guardrail ativado: %s -> %s", rule_name, action_taken)

    def log_model_inference(
        self,
        model_name: str,
        prompt_length: int,
        response_length: int,
        latency_ms: float,
    ) -> None:
        """Registra métricas de inferência do modelo."""
        entry = {
            "event_type": "model_inference",
            "model_name": model_name,
            "prompt_tokens_approx": prompt_length // 4,
            "response_tokens_approx": response_length // 4,
            "latency_ms": latency_ms,
        }
        self._write_entry(entry)


def setup_logging(log_level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """Configura logging global da aplicação."""
    log_dir_path = Path(log_dir or os.environ.get("LOG_DIR", "./logs"))
    log_dir_path.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir_path / "application.log", encoding="utf-8"),
        ],
    )
