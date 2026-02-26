"""
Guardrails de segurança para o assistente médico.

Implementa filtros de input e output para garantir que o sistema
não ultrapasse seus limites de atuação (nunca prescrever diretamente,
sempre citar fontes, incluir disclaimer, etc.).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from medical_assistant.domain.entities.medical_response import MedicalResponse

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


class Guardrails:
    """
    Guardrails de segurança para filtrar inputs e outputs do assistente.

    Carrega regras de configs/guardrails_config.yaml e aplica validações
    automaticamente nas interações.
    """

    def __init__(self, config_path: str | Path | None = None):
        config_path = Path(config_path or CONFIGS_DIR / "guardrails_config.yaml")
        self.config = self._load_config(config_path)

        # Compilar regex de input bloqueado
        self._blocked_input_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.get("input_filters", {}).get("blocked_patterns", [])
        ]

        # Compilar regex de output proibido
        self._prohibited_output = []
        for rule in self.config.get("output_filters", {}).get("prohibited_actions", []):
            self._prohibited_output.append({
                "pattern": re.compile(rule["pattern"], re.IGNORECASE),
                "message": rule["message"],
            })

        # Compilar regex de escalação
        self._escalation_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.get("output_filters", {}).get("escalation_triggers", [])
        ]

        logger.info(
            "Guardrails carregados: %d filtros de input, %d filtros de output, %d triggers de escalação",
            len(self._blocked_input_patterns),
            len(self._prohibited_output),
            len(self._escalation_patterns),
        )

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Carrega configuração de guardrails."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        logger.warning("Config de guardrails não encontrado: %s", path)
        return {}

    def validate_input(self, text: str) -> tuple[bool, str]:
        """
        Valida input do usuário.

        Returns:
            Tupla (is_valid, rejection_message)
        """
        # Verificar tamanho
        max_len = self.config.get("input_filters", {}).get("max_input_length", 5000)
        if len(text) > max_len:
            return False, f"Input excede o limite de {max_len} caracteres."

        # Verificar padrões bloqueados
        for pattern in self._blocked_input_patterns:
            if pattern.search(text):
                return False, "Esta pergunta está fora do escopo do assistente médico."

        return True, ""

    def apply(
        self, response: MedicalResponse
    ) -> tuple[MedicalResponse, list[str]]:
        """
        Aplica guardrails na resposta gerada.

        Args:
            response: Resposta da LLM

        Returns:
            Tupla (resposta_modificada, guardrails_ativados)
        """
        triggered: list[str] = []

        # Verificar padrões proibidos no output
        for rule in self._prohibited_output:
            if rule["pattern"].search(response.response_text):
                triggered.append(rule["message"])
                # Adicionar aviso na resposta
                response.response_text += f"\n\n⚠️ {rule['message']}"

        # Verificar confiança mínima
        min_confidence = self.config.get("safety", {}).get("min_confidence_score", 0.3)
        if response.confidence.value < min_confidence:
            low_msg = self.config.get("safety", {}).get(
                "low_confidence_message",
                "Confiança insuficiente. Consulte o especialista."
            )
            triggered.append("low_confidence")
            response.response_text = low_msg.strip()

        # Verificar triggers de escalação
        for pattern in self._escalation_patterns:
            if pattern.search(response.response_text):
                triggered.append("escalation_required")
                response.response_text += (
                    "\n\n🚨 ATENÇÃO: Esta resposta contém informações que requerem "
                    "validação imediata pela equipe médica."
                )
                break  # Uma vez é suficiente

        # Garantir que fontes estão presentes (se exigido)
        require_sources = self.config.get("output_filters", {}).get("require_sources", True)
        if require_sources and not response.sources:
            triggered.append("missing_sources")

        # Garantir disclaimer
        if response.disclaimer not in response.response_text:
            response.response_text += f"\n\n{response.disclaimer}"

        # Registrar nos metadados
        response.guardrails_triggered = triggered

        if triggered:
            logger.info("Guardrails ativados: %s", triggered)

        return response, triggered
