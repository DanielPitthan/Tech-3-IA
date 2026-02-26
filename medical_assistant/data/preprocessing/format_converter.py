"""
Conversor de formato para datasets de fine-tuning.

Converte datasets processados para formatos compatíveis com
HuggingFace datasets / SFTTrainer (Alpaca, ShareGPT, etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def to_alpaca_format(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Converte para formato Alpaca (instruction, input, output).

    Formato Alpaca:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    alpaca_samples = []
    for s in samples:
        alpaca_samples.append({
            "instruction": s.get("instruction", ""),
            "input": s.get("input", ""),
            "output": s.get("output", ""),
        })
    return alpaca_samples


def to_chatml_format(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Converte para formato ChatML (messages array).

    Formato ChatML:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    chatml_samples = []
    for s in samples:
        messages = [
            {"role": "system", "content": s.get("instruction", "")},
            {"role": "user", "content": s.get("input", "")},
            {"role": "assistant", "content": s.get("output", "")},
        ]
        chatml_samples.append({"messages": messages})
    return chatml_samples


def to_text_format(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Converte para formato de texto simples (campo 'text').
    Usado diretamente pelo SFTTrainer com dataset_text_field="text".
    """
    text_samples = []
    for s in samples:
        if "text" in s:
            text_samples.append({"text": s["text"]})
        else:
            text = (
                f"### Instrução:\n{s.get('instruction', '')}\n\n"
                f"### Entrada:\n{s.get('input', '')}\n\n"
                f"### Resposta:\n{s.get('output', '')}"
            )
            text_samples.append({"text": text})
    return text_samples


def load_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """Carrega arquivo JSONL."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_jsonl(samples: list[dict[str, Any]], filepath: str | Path) -> None:
    """Salva lista de dicionários como JSONL."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Salvo %d amostras em: %s", len(samples), filepath)


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
    target_format: str = "text",
) -> list[dict[str, Any]]:
    """
    Converte dataset de um formato para outro.

    Args:
        input_path: Caminho do JSONL de entrada
        output_path: Caminho do JSONL de saída
        target_format: 'alpaca', 'chatml', ou 'text'
    """
    samples = load_jsonl(input_path)

    converters = {
        "alpaca": to_alpaca_format,
        "chatml": to_chatml_format,
        "text": to_text_format,
    }

    if target_format not in converters:
        raise ValueError(f"Formato desconhecido: {target_format}. Use: {list(converters.keys())}")

    converted = converters[target_format](samples)
    save_jsonl(converted, output_path)

    logger.info(
        "Convertido %d amostras de %s para formato '%s' em %s",
        len(converted),
        input_path,
        target_format,
        output_path,
    )
    return converted
