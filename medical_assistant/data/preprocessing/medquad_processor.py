"""
Processador do dataset MedQuAD / LiveQA-Med.

Parseia o CSV de respostas médicas e o arquivo de julgamentos de relevância,
filtrando respostas de alta qualidade para uso como base de conhecimento (RAG)
e dados adicionais de fine-tuning.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_answer_field(answer_text: str) -> dict[str, str]:
    """
    Extrai pergunta e resposta do campo 'Answer' do MedQuAD.

    O formato típico é:
        <pergunta original>
        <URL fonte>
        <resposta completa>
    """
    lines = answer_text.strip().split("\n")

    question = ""
    url = ""
    answer_lines: list[str] = []

    state = "question"
    for line in lines:
        line_s = line.strip()
        if state == "question" and line_s:
            if line_s.startswith("http"):
                url = line_s
                state = "answer"
            elif not question:
                question = line_s
            else:
                # Pode ser continuação da pergunta ou início da resposta
                if re.match(r"https?://", line_s):
                    url = line_s
                    state = "answer"
                else:
                    # Considerar como parte da pergunta
                    question += " " + line_s
        elif state == "answer":
            answer_lines.append(line_s)
        elif line_s.startswith("http"):
            url = line_s
            state = "answer"

    # Se não encontrou URL, tudo após a primeira linha é resposta
    if not url and len(lines) > 1:
        question = lines[0].strip()
        answer_lines = [l.strip() for l in lines[1:]]

    answer_body = "\n".join(answer_lines).strip()

    return {
        "question": question,
        "url": url,
        "answer": answer_body,
    }


def load_medquad_answers(csv_path: str | Path) -> list[dict[str, Any]]:
    """Carrega e parseia o CSV do MedQuAD."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    records: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_id = row.get("AnswerID", "").strip()
            answer_raw = row.get("Answer", "").strip()
            if not answer_id or not answer_raw:
                continue

            parsed = parse_answer_field(answer_raw)
            records.append({
                "answer_id": answer_id,
                "question": parsed["question"],
                "url": parsed["url"],
                "answer": parsed["answer"],
                "raw_text": answer_raw,
            })

    logger.info("MedQuAD carregado: %d respostas", len(records))
    return records


def load_relevance_judgments(qrels_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """
    Carrega os julgamentos de relevância do LiveQA-Med.

    Formato: question_id  relevance_label  answer_id
    Labels: 1=Incorrect, 2=Related, 3=Incomplete, 4=Exact
    """
    qrels_path = Path(qrels_path)
    if not qrels_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {qrels_path}")

    judgments: dict[str, list[dict[str, Any]]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # Ignorar cabeçalho se existir
            try:
                q_id = parts[0]
                relevance = int(parts[1].split("-")[0]) if "-" in parts[1] else int(parts[1])
                a_id = parts[2]
            except (ValueError, IndexError):
                continue

            if q_id not in judgments:
                judgments[q_id] = []
            judgments[q_id].append({
                "answer_id": a_id,
                "relevance": relevance,
            })

    total = sum(len(v) for v in judgments.values())
    logger.info("Julgamentos carregados: %d julgamentos para %d perguntas", total, len(judgments))
    return judgments


def filter_high_quality_answers(
    answers: list[dict[str, Any]],
    judgments: dict[str, list[dict[str, Any]]],
    min_relevance: int = 3,
) -> list[dict[str, Any]]:
    """
    Filtra respostas com score de relevância >= min_relevance.

    Args:
        answers: Lista de respostas do MedQuAD
        judgments: Dicionário de julgamentos por question_id
        min_relevance: Score mínimo (3=Incomplete, 4=Exact)

    Returns:
        Lista de respostas de alta qualidade
    """
    # Criar mapa de answer_id -> melhor relevância
    answer_relevance: dict[str, int] = {}
    for q_judgments in judgments.values():
        for j in q_judgments:
            a_id = j["answer_id"]
            rel = j["relevance"]
            if a_id not in answer_relevance or rel > answer_relevance[a_id]:
                answer_relevance[a_id] = rel

    filtered = []
    for ans in answers:
        a_id = ans["answer_id"]
        rel = answer_relevance.get(a_id, 0)
        if rel >= min_relevance:
            ans["relevance_score"] = rel
            filtered.append(ans)

    logger.info(
        "Filtragem de qualidade: %d/%d respostas com relevância >= %d",
        len(filtered),
        len(answers),
        min_relevance,
    )
    return filtered


def format_medquad_for_training(
    answers: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Formata respostas do MedQuAD para formato de instrução (fine-tuning).
    """
    samples = []
    for ans in answers:
        if not ans["question"] or not ans["answer"]:
            continue

        instruction = (
            "Você é um assistente médico especializado. Responda à pergunta médica "
            "a seguir de forma clara, precisa e baseada em evidências. Cite a fonte "
            "da informação quando disponível."
        )

        input_text = f"Pergunta: {ans['question']}"

        source_info = f"\n\nFonte: {ans['url']}" if ans.get("url") else ""
        output_text = f"{ans['answer']}{source_info}"

        full_text = (
            f"### Instrução:\n{instruction}\n\n"
            f"### Entrada:\n{input_text}\n\n"
            f"### Resposta:\n{output_text}"
        )

        samples.append({
            "answer_id": ans["answer_id"],
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "text": full_text,
            "source": "medquad",
            "relevance_score": ans.get("relevance_score", 0),
        })

    logger.info("MedQuAD formatado: %d amostras de instrução", len(samples))
    return samples


def format_medquad_for_rag(answers: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Formata respostas do MedQuAD para indexação no vector store (RAG).

    Cada resposta se torna um documento com metadata.
    """
    documents = []
    for ans in answers:
        if not ans["answer"]:
            continue

        # Conteúdo do documento para embedding
        content = f"Pergunta: {ans['question']}\n\nResposta: {ans['answer']}"

        documents.append({
            "content": content,
            "metadata": {
                "answer_id": ans["answer_id"],
                "source_url": ans.get("url", ""),
                "source": "MedQuAD",
                "relevance_score": ans.get("relevance_score", 0),
            },
        })

    logger.info("MedQuAD para RAG: %d documentos preparados", len(documents))
    return documents


def process_medquad(
    csv_path: str | Path,
    qrels_path: str | Path,
    output_training_path: str | Path | None = None,
    output_rag_path: str | Path | None = None,
    min_relevance: int = 3,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Pipeline completo de processamento do MedQuAD.

    Returns:
        Tupla (amostras_de_treinamento, documentos_rag)
    """
    # Carregar dados
    answers = load_medquad_answers(csv_path)
    judgments = load_relevance_judgments(qrels_path)

    # Filtrar por qualidade
    high_quality = filter_high_quality_answers(answers, judgments, min_relevance)

    # Formatar para treinamento
    training_samples = format_medquad_for_training(high_quality)

    # Formatar para RAG (usar todas as respostas, não apenas alta qualidade)
    rag_documents = format_medquad_for_rag(answers)

    # Salvar training
    if output_training_path:
        output_training_path = Path(output_training_path)
        output_training_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_training_path, "w", encoding="utf-8") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Training data salvo em: %s", output_training_path)

    # Salvar RAG docs
    if output_rag_path:
        output_rag_path = Path(output_rag_path)
        output_rag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_rag_path, "w", encoding="utf-8") as f:
            for doc in rag_documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info("RAG docs salvos em: %s", output_rag_path)

    return training_samples, rag_documents


class MedQuADProcessor:
    """Wrapper orientado a objetos para o pipeline de processamento MedQuAD."""

    def __init__(self, min_relevance: int = 3) -> None:
        self.min_relevance = min_relevance

    def load_and_process(
        self, csv_path: str | Path, qrels_path: str | Path
    ) -> list[dict[str, Any]]:
        """Carrega, filtra e retorna respostas de alta qualidade."""
        answers = load_medquad_answers(csv_path)
        judgments = load_relevance_judgments(qrels_path)
        return filter_high_quality_answers(answers, judgments, self.min_relevance)

    def save_training_data(
        self, entries: list[dict[str, Any]], output_path: str | Path
    ) -> None:
        """Formata e salva dados para fine-tuning."""
        samples = format_medquad_for_training(entries)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Training data MedQuAD salvo em: %s (%d amostras)", output_path, len(samples))

    def save_rag_documents(
        self, entries: list[dict[str, Any]], output_path: str | Path
    ) -> None:
        """Formata e salva documentos para RAG."""
        documents = format_medquad_for_rag(entries)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logger.info("RAG docs MedQuAD salvos em: %s (%d docs)", output_path, len(documents))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BASE = Path(__file__).resolve().parents[3]
    ds_dir = BASE / "DataSets" / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers"
    ds_dir = ds_dir / "QA-TestSet-LiveQA-Med-Qrels-2479-Answers"

    process_medquad(
        csv_path=ds_dir / "All-2479-Answers-retrieved-from-MedQuAD.csv",
        qrels_path=ds_dir / "All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt",
        output_training_path=BASE / "data" / "processed" / "medquad_instructions.jsonl",
        output_rag_path=BASE / "data" / "processed" / "medquad_rag_docs.jsonl",
    )
