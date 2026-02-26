"""
Divisor de dataset para fine-tuning.

Separa os dados em conjuntos de treino/validação/teste,
usando o test_ground_truth.json do PubMedQA como referência
para o split de teste fixo.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_test_ids(test_ground_truth_path: str | Path) -> set[str]:
    """Carrega os PMIDs do conjunto de teste oficial."""
    path = Path(test_ground_truth_path)
    with open(path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_ids = set(test_data.keys())
    logger.info("IDs de teste carregados: %d", len(test_ids))
    return test_ids


def split_dataset(
    samples: list[dict[str, Any]],
    test_ids: set[str] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    id_field: str = "pmid",
) -> dict[str, list[dict[str, Any]]]:
    """
    Divide o dataset em train/val/test.

    Args:
        samples: Lista de amostras processadas
        test_ids: IDs pré-definidos para o conjunto de teste
        val_ratio: Proporção do conjunto de validação (sobre o restante)
        seed: Semente aleatória para reprodutibilidade
        id_field: Campo usado como identificador

    Returns:
        Dicionário com chaves 'train', 'val', 'test'
    """
    random.seed(seed)

    if test_ids:
        test_set = [s for s in samples if s.get(id_field) in test_ids]
        remaining = [s for s in samples if s.get(id_field) not in test_ids]
    else:
        # Se não há IDs de teste, usar 20% como teste
        random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)
        remaining = samples[:split_idx]
        test_set = samples[split_idx:]

    # Split estratificado por label no remaining
    label_groups: dict[str, list[dict[str, Any]]] = {}
    for s in remaining:
        label = s.get("label", "unknown")
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(s)

    train_set: list[dict[str, Any]] = []
    val_set: list[dict[str, Any]] = []

    for label, group in label_groups.items():
        random.shuffle(group)
        val_size = max(1, int(len(group) * val_ratio))
        val_set.extend(group[:val_size])
        train_set.extend(group[val_size:])

    # Embaralhar
    random.shuffle(train_set)
    random.shuffle(val_set)

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_set),
        len(val_set),
        len(test_set),
    )

    # Distribuição por label
    for name, split in [("train", train_set), ("val", val_set), ("test", test_set)]:
        counts: dict[str, int] = {}
        for s in split:
            lbl = s.get("label", "unknown")
            counts[lbl] = counts.get(lbl, 0) + 1
        logger.info("  %s: %s", name, counts)

    return {"train": train_set, "val": val_set, "test": test_set}


def save_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    prefix: str = "dataset",
) -> None:
    """Salva os splits em arquivos JSONL separados."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        filepath = output_dir / f"{prefix}_{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("Salvo %s: %d amostras em %s", split_name, len(samples), filepath)


def merge_datasets(
    *datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mescla múltiplos datasets processados em um único."""
    merged: list[dict[str, Any]] = []
    for ds in datasets:
        merged.extend(ds)
    logger.info("Datasets mesclados: %d amostras totais", len(merged))
    return merged


class DatasetSplitter:
    """Wrapper orientado a objetos para o pipeline de split de datasets."""

    def __init__(self, val_ratio: float = 0.15, seed: int = 42) -> None:
        self.val_ratio = val_ratio
        self.seed = seed

    def split(
        self,
        processed_jsonl: str | Path,
        ground_truth_path: str | Path,
        output_dir: str | Path,
        prefix: str = "dataset",
    ) -> dict[str, int]:
        """
        Carrega amostras de um JSONL, faz split e salva.

        Returns:
            Dicionário com contagem de amostras: {'train': N, 'val': N, 'test': N}
        """
        processed_jsonl = Path(processed_jsonl)
        if not processed_jsonl.exists():
            logger.warning("Arquivo %s não encontrado, split ignorado.", processed_jsonl)
            return {"train": 0, "val": 0, "test": 0}

        # Carregar amostras
        samples: list[dict[str, Any]] = []
        with open(processed_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Carregar IDs de teste
        test_ids = load_test_ids(ground_truth_path)

        # Fazer split
        splits = split_dataset(
            samples, test_ids=test_ids, val_ratio=self.val_ratio, seed=self.seed
        )

        # Salvar
        save_splits(splits, output_dir=output_dir, prefix=prefix)

        return {k: len(v) for k, v in splits.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BASE = Path(__file__).resolve().parents[3]

    # Carregar dados processados
    from medical_assistant.data.preprocessing.pubmedqa_processor import process_pubmedqa

    pubmedqa_samples = process_pubmedqa(
        filepath=BASE / "DataSets" / "ori_pqal.json",
    )

    # Carregar IDs de teste
    test_ids = load_test_ids(BASE / "DataSets" / "test_ground_truth.json")

    # Split
    splits = split_dataset(pubmedqa_samples, test_ids=test_ids)

    # Salvar
    save_splits(splits, output_dir=BASE / "data" / "processed", prefix="pubmedqa")
