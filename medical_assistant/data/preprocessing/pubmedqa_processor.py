"""
Processador do dataset PubMedQA (ori_pqal.json).

Converte o formato original (dicionário de PMIDs) para o formato de instrução
(prompt → completion) compatível com SFTTrainer para fine-tuning.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_pubmedqa(filepath: str | Path) -> dict[str, dict[str, Any]]:
    """Carrega o dataset PubMedQA a partir do JSON original."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("PubMedQA carregado: %d registros", len(data))
    return data


def anonymize_text(text: str) -> str:
    """
    Remove informações pessoais identificáveis (PHI) do texto.

    Aplica regex para detectar e anonimizar:
    - Nomes próprios em contexto de paciente
    - Números de telefone
    - CPFs
    - Emails
    - Endereços IP
    """
    # CPF brasileiro (XXX.XXX.XXX-XX)
    text = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF_ANONIMIZADO]", text)

    # Telefones brasileiros
    text = re.sub(
        r"\b(?:\+55\s?)?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b",
        "[TELEFONE_ANONIMIZADO]",
        text,
    )

    # Emails
    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "[EMAIL_ANONIMIZADO]",
        text,
    )

    return text


def format_context(contexts: list[str], labels: list[str]) -> str:
    """Formata os contextos do PubMedQA com seus rótulos."""
    parts = []
    for label, context in zip(labels, contexts):
        parts.append(f"[{label}] {context}")
    return "\n".join(parts)


DECISION_MAP = {
    "yes": "Sim",
    "no": "Não",
    "maybe": "Talvez",
}


def format_instruction_sample(
    pmid: str,
    record: dict[str, Any],
    include_meshes: bool = True,
) -> dict[str, str]:
    """
    Converte um registro PubMedQA para formato de instrução (Alpaca-style).

    Returns:
        Dicionário com campos: instruction, input, output, text (concatenado)
    """
    question = record["QUESTION"]
    contexts = record.get("CONTEXTS", [])
    labels = record.get("LABELS", [])
    long_answer = record.get("LONG_ANSWER", "")
    decision = record.get("final_decision", "")
    meshes = record.get("MESHES", [])

    # Formatar contexto
    context_text = format_context(contexts, labels)

    # Anonimizar
    context_text = anonymize_text(context_text)
    long_answer = anonymize_text(long_answer)

    # Montar instrução
    instruction = (
        "Você é um assistente médico especializado. Com base no contexto científico "
        "fornecido, responda à pergunta clínica a seguir. Forneça sua decisão "
        "(Sim/Não/Talvez) e uma justificativa detalhada baseada nas evidências."
    )

    # Input com contexto
    input_parts = [f"Pergunta: {question}", "", f"Contexto científico:\n{context_text}"]
    if include_meshes and meshes:
        input_parts.append(f"\nTermos MeSH: {', '.join(meshes)}")
    input_text = "\n".join(input_parts)

    # Output esperado
    decision_pt = DECISION_MAP.get(decision, decision)
    output_text = f"Decisão: {decision_pt}\n\nJustificativa: {long_answer}"

    # Texto completo para SFTTrainer
    full_text = (
        f"### Instrução:\n{instruction}\n\n"
        f"### Entrada:\n{input_text}\n\n"
        f"### Resposta:\n{output_text}"
    )

    return {
        "pmid": pmid,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "text": full_text,
        "label": decision,
    }


def process_pubmedqa(
    filepath: str | Path,
    output_path: str | Path | None = None,
    include_meshes: bool = True,
) -> list[dict[str, str]]:
    """
    Pipeline completo de processamento do PubMedQA.

    Args:
        filepath: Caminho para ori_pqal.json
        output_path: Caminho para salvar o dataset processado (JSONL)
        include_meshes: Incluir termos MeSH no input

    Returns:
        Lista de amostras no formato de instrução
    """
    raw_data = load_pubmedqa(filepath)

    samples = []
    for pmid, record in raw_data.items():
        # Validar campos obrigatórios
        if not record.get("QUESTION") or not record.get("final_decision"):
            logger.warning("Registro %s sem campos obrigatórios, ignorando.", pmid)
            continue

        sample = format_instruction_sample(pmid, record, include_meshes)
        samples.append(sample)

    logger.info(
        "PubMedQA processado: %d amostras formatadas (de %d registros)",
        len(samples),
        len(raw_data),
    )

    # Estatísticas de distribuição
    label_counts: dict[str, int] = {}
    for s in samples:
        lbl = s["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    logger.info("Distribuição de labels: %s", label_counts)

    # Salvar se output_path fornecido
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Dataset salvo em: %s", output_path)

    return samples


class PubMedQAProcessor:
    """Wrapper orientado a objetos para o pipeline de processamento PubMedQA."""

    def __init__(self, include_meshes: bool = True) -> None:
        self.include_meshes = include_meshes

    def load_and_process(self, filepath: str | Path) -> list[dict[str, str]]:
        """Carrega e processa o PubMedQA, retornando as amostras formatadas."""
        return process_pubmedqa(filepath, include_meshes=self.include_meshes)

    def save_jsonl(self, samples: list[dict[str, str]], output_path: str | Path) -> None:
        """Salva amostras em JSONL."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Dataset PubMedQA salvo em: %s (%d amostras)", output_path, len(samples))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BASE = Path(__file__).resolve().parents[3]
    process_pubmedqa(
        filepath=BASE / "DataSets" / "ori_pqal.json",
        output_path=BASE / "data" / "processed" / "pubmedqa_instructions.jsonl",
    )
