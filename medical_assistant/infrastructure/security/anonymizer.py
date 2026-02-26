"""
Anonimizador de dados de pacientes (PHI - Protected Health Information).
"""

from __future__ import annotations

import re


class Anonymizer:
    """Anonimiza informações pessoais identificáveis (PHI) em textos."""

    PATTERNS = [
        # CPF (XXX.XXX.XXX-XX)
        (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[CPF_ANONIMIZADO]"),
        # RG
        (re.compile(r"\b\d{2}\.\d{3}\.\d{3}-\d{1}\b"), "[RG_ANONIMIZADO]"),
        # Telefone brasileiro
        (
            re.compile(r"\b(?:\+55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),
            "[TELEFONE_ANONIMIZADO]",
        ),
        # Email
        (
            re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
            "[EMAIL_ANONIMIZADO]",
        ),
        # Data de nascimento (DD/MM/AAAA)
        (
            re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
            "[DATA_ANONIMIZADA]",
        ),
        # CEP
        (re.compile(r"\b\d{5}-\d{3}\b"), "[CEP_ANONIMIZADO]"),
    ]

    def anonymize(self, text: str) -> str:
        """Aplica todos os padrões de anonimização ao texto."""
        for pattern, replacement in self.PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def has_phi(self, text: str) -> bool:
        """Verifica se o texto contém PHI."""
        for pattern, _ in self.PATTERNS:
            if pattern.search(text):
                return True
        return False

    def detect_phi(self, text: str) -> list[dict[str, str]]:
        """Detecta e retorna todas as ocorrências de PHI encontradas."""
        found: list[dict[str, str]] = []
        for pattern, replacement in self.PATTERNS:
            for match in pattern.finditer(text):
                found.append({
                    "type": replacement.strip("[]"),
                    "value": match.group(),
                    "position": f"{match.start()}-{match.end()}",
                })
        return found
