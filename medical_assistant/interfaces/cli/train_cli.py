"""
CLI de Treinamento — Pré-processamento de dados e fine-tuning.

Comandos:
    medassist train preprocess   — Pré-processar datasets
    medassist train finetune     — Executar fine-tuning QLoRA
    medassist train generate-patients — Gerar pacientes sintéticos
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

train_app = typer.Typer(
    name="train",
    help="Treinamento e pré-processamento de dados",
    no_args_is_help=True,
)
console = Console()


@train_app.command()
def preprocess(
    pubmedqa_file: str = typer.Option(
        "DataSets/ori_pqal.json",
        "--pubmedqa",
        help="Caminho para ori_pqal.json (PubMedQA)",
    ),
    medquad_csv: str = typer.Option(
        "DataSets/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/"
        "QA-TestSet-LiveQA-Med-Qrels-2479-Answers/"
        "All-2479-Answers-retrieved-from-MedQuAD.csv",
        "--medquad",
        help="Caminho para o CSV MedQuAD",
    ),
    medquad_qrels: str = typer.Option(
        "DataSets/QA-TestSet-LiveQA-Med-Qrels-2479-Answers/"
        "QA-TestSet-LiveQA-Med-Qrels-2479-Answers/"
        "All-qrels_LiveQAMed2017-TestQuestions_2479_Judged-Answers.txt",
        "--qrels",
        help="Caminho para qrels de relevância",
    ),
    ground_truth: str = typer.Option(
        "DataSets/test_ground_truth.json",
        "--ground-truth",
        help="Caminho para test_ground_truth.json",
    ),
    output_dir: str = typer.Option("data/processed", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Pré-processar PubMedQA e MedQuAD para treinamento."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. PubMedQA
        task1 = progress.add_task("Processando PubMedQA...", total=None)
        try:
            from medical_assistant.data.preprocessing.pubmedqa_processor import PubMedQAProcessor

            processor = PubMedQAProcessor()
            entries = processor.load_and_process(pubmedqa_file)
            pubmed_output = str(output_path / "pubmedqa_processed.jsonl")
            processor.save_jsonl(entries, pubmed_output)
            progress.update(task1, description=f"✓ PubMedQA: {len(entries)} entradas")
        except Exception as e:
            progress.update(task1, description=f"✗ PubMedQA: {e}")
            console.print(f"[red]Erro PubMedQA: {e}[/red]")

        # 2. MedQuAD
        task2 = progress.add_task("Processando MedQuAD...", total=None)
        try:
            from medical_assistant.data.preprocessing.medquad_processor import MedQuADProcessor

            med_processor = MedQuADProcessor()
            med_entries = med_processor.load_and_process(medquad_csv, medquad_qrels)

            # Salvar para treinamento
            train_output = str(output_path / "medquad_train.jsonl")
            med_processor.save_training_data(med_entries, train_output)

            # Salvar para RAG
            rag_output = str(output_path / "medquad_rag.jsonl")
            med_processor.save_rag_documents(med_entries, rag_output)

            progress.update(task2, description=f"✓ MedQuAD: {len(med_entries)} entradas")
        except Exception as e:
            progress.update(task2, description=f"✗ MedQuAD: {e}")
            console.print(f"[red]Erro MedQuAD: {e}[/red]")

        # 3. Split do dataset
        task3 = progress.add_task("Dividindo datasets...", total=None)
        try:
            from medical_assistant.data.preprocessing.dataset_splitter import DatasetSplitter

            splitter = DatasetSplitter()
            combined = str(output_path / "pubmedqa_processed.jsonl")
            splits = splitter.split(combined, ground_truth, str(output_path))
            progress.update(
                task3,
                description=f"✓ Split: train={splits.get('train', 0)}, "
                            f"val={splits.get('val', 0)}, test={splits.get('test', 0)}",
            )
        except Exception as e:
            progress.update(task3, description=f"✗ Split: {e}")
            console.print(f"[red]Erro Split: {e}[/red]")

    console.print("\n[green]✓ Pré-processamento concluído![/green]")
    console.print(f"Dados em: [cyan]{output_dir}[/cyan]")


@train_app.command()
def finetune(
    config: str = typer.Option("configs/qlora_config.yaml", "--config", "-c"),
    model_config: str = typer.Option("configs/model_config.yaml", "--model-config"),
    train_file: str = typer.Option("data/processed/train.jsonl", "--train-file"),
    val_file: str = typer.Option("data/processed/val.jsonl", "--val-file"),
    output_dir: str = typer.Option("models/falcon-7b-qlora", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Executar fine-tuning QLoRA do Falcon-7B."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    console.print(Panel(
        f"[bold]Fine-tuning QLoRA[/bold]\n"
        f"Config: {config}\n"
        f"Modelo: {model_config}\n"
        f"Treino: {train_file}\n"
        f"Saída: {output_dir}",
        title="🔧 Fine-tuning",
        border_style="blue",
    ))

    try:
        from medical_assistant.infrastructure.llm.model_config import (
            ModelConfig,
            QuantizationConfig,
            LoraConfig,
            TrainingConfig,
            SFTConfig,
        )
        from medical_assistant.infrastructure.llm.falcon_qlora_trainer import FalconQLoRATrainer

        # Carregar configurações
        mc = ModelConfig.from_yaml(model_config)
        qc = QuantizationConfig.from_yaml(model_config)
        lc = LoraConfig.from_yaml(config)
        tc = TrainingConfig.from_yaml(config)
        sc = SFTConfig.from_yaml(config)

        tc.output_dir = output_dir

        trainer = FalconQLoRATrainer(
            model_config=mc,
            quant_config=qc,
            lora_config=lc,
            training_config=tc,
            sft_config=sc,
        )

        console.print("[dim]Iniciando pipeline de fine-tuning...[/dim]")
        trainer.run_pipeline(train_file, val_file)

        console.print(f"\n[green]✓ Fine-tuning concluído! Modelo salvo em: {output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Erro no fine-tuning: {e}[/red]")
        raise typer.Exit(1)


@train_app.command(name="generate-patients")
def generate_patients(
    count: int = typer.Option(20, "--count", "-n", help="Número de pacientes"),
    output_dir: str = typer.Option("data/patients", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Gerar pacientes sintéticos para testes."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    import json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from medical_assistant.data.synthetic.synthetic_patients import SyntheticPatientGenerator

        generator = SyntheticPatientGenerator()
        patients = generator.generate_batch(count)

        for patient in patients:
            filepath = output_path / f"patient_{patient['id']}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(patient, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ {len(patients)} pacientes gerados em {output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


@train_app.command(name="index-knowledge")
def index_knowledge(
    rag_file: str = typer.Option("data/processed/medquad_rag.jsonl", "--input", "-i"),
    chroma_dir: str = typer.Option("chroma_db", "--chroma-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Indexar base de conhecimento no ChromaDB para RAG."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        import json
        from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore

        # Carregar documentos
        documents: list[str] = []
        metadatas: list[dict] = []

        with open(rag_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                documents.append(entry.get("content", ""))
                metadatas.append(entry.get("metadata", {}))

        console.print(f"[dim]Carregados {len(documents)} documentos[/dim]")

        # Indexar
        store = MedicalVectorStore(persist_directory=chroma_dir)
        store.add_documents(documents, metadatas)

        console.print(f"[green]✓ {len(documents)} documentos indexados no ChromaDB[/green]")
        console.print(f"[dim]Persistent directory: {chroma_dir}[/dim]")

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


# Necessário para importação em main.py
from rich.panel import Panel
