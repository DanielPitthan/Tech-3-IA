"""
CLI de Avaliação — Benchmark de métricas e LLM-as-judge.

Comandos:
    medassist evaluate benchmark  — Benchmark de métricas quantitativas
    medassist evaluate judge      — Avaliação qualitativa com LLM-as-judge
    medassist evaluate report     — Gerar relatório de avaliação
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

evaluate_app = typer.Typer(
    name="evaluate",
    help="Avaliação e benchmark do modelo",
    no_args_is_help=True,
)
console = Console()


@evaluate_app.command()
def benchmark(
    test_file: str = typer.Option("data/processed/test.jsonl", "--test-file", "-t"),
    model_dir: str = typer.Option("models/falcon-7b-qlora", "--model-dir", "-m"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n"),
    output_dir: str = typer.Option("evaluation_results", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Executar benchmark de métricas quantitativas."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    console.print(Panel(
        f"[bold]Benchmark Quantitativo[/bold]\n"
        f"Dataset: {test_file}\n"
        f"Modelo: {model_dir}\n"
        f"Amostras: {max_samples or 'todas'}",
        title="📊 Benchmark",
        border_style="blue",
    ))

    try:
        from medical_assistant.infrastructure.llm.falcon_model import FalconModelAdapter
        from medical_assistant.evaluation.benchmark import BenchmarkRunner

        base_model = os.getenv("BASE_MODEL_NAME", "tiiuae/falcon-7b-instruct")

        console.print("[dim]Carregando modelo...[/dim]")
        llm_service = FalconModelAdapter(
            model_name=base_model,
            adapter_path=model_dir if Path(model_dir).exists() else None,
        )
        llm_service.load()

        runner = BenchmarkRunner(llm_service=llm_service, output_dir=output_dir)
        result = runner.run(
            test_file=test_file,
            model_name="falcon-7b-qlora",
            max_samples=max_samples,
        )

        _display_benchmark_result(result)

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


@evaluate_app.command()
def judge(
    test_file: str = typer.Option("data/processed/test.jsonl", "--test-file", "-t"),
    model_dir: str = typer.Option("models/falcon-7b-qlora", "--model-dir", "-m"),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model"),
    max_samples: int = typer.Option(50, "--max-samples", "-n"),
    output_dir: str = typer.Option("evaluation_results", "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Avaliação qualitativa com LLM-as-judge (requer OPENAI_API_KEY)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]OPENAI_API_KEY não configurada. Defina no .env[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]LLM-as-Judge[/bold]\n"
        f"Judge: {judge_model}\n"
        f"Amostras: {max_samples}",
        title="🧑‍⚖️ Avaliação Qualitativa",
        border_style="magenta",
    ))

    try:
        from medical_assistant.infrastructure.llm.falcon_model import FalconModelAdapter
        from medical_assistant.evaluation.benchmark import BenchmarkRunner
        from medical_assistant.evaluation.llm_judge import LLMJudge

        base_model = os.getenv("BASE_MODEL_NAME", "tiiuae/falcon-7b-instruct")

        console.print("[dim]Carregando modelo...[/dim]")
        llm_service = FalconModelAdapter(
            model_name=base_model,
            adapter_path=model_dir if Path(model_dir).exists() else None,
        )
        llm_service.load()

        llm_judge = LLMJudge(model=judge_model, api_key=api_key)

        runner = BenchmarkRunner(
            llm_service=llm_service,
            judge=llm_judge,
            output_dir=output_dir,
        )

        result = runner.run(
            test_file=test_file,
            model_name="falcon-7b-qlora",
            max_samples=max_samples,
            run_judge=True,
            judge_max_samples=max_samples,
        )

        _display_benchmark_result(result)

        if result.judge_report:
            console.print(result.judge_report.summary())

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


@evaluate_app.command()
def report(
    results_dir: str = typer.Option("evaluation_results", "--input", "-i"),
) -> None:
    """Gerar relatório consolidado de avaliação."""
    results_path = Path(results_dir)

    if not results_path.exists():
        console.print(f"[red]Diretório não encontrado: {results_dir}[/red]")
        raise typer.Exit(1)

    json_files = sorted(results_path.glob("benchmark_*.json"))
    if not json_files:
        console.print("[yellow]Nenhum resultado de benchmark encontrado.[/yellow]")
        return

    console.print(Panel(
        f"[bold]Relatório de Avaliação[/bold]\n"
        f"Resultados encontrados: {len(json_files)}",
        title="📋 Relatório",
        border_style="cyan",
    ))

    table = Table(title="Resultados de Benchmark")
    table.add_column("Modelo", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 (macro)", justify="right")
    table.add_column("Exact Match", justify="right")
    table.add_column("Token F1", justify="right")
    table.add_column("Judge", justify="right")
    table.add_column("Amostras", justify="right")

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            model = data.get("model_name", "?")
            cm = data.get("classification_metrics", {})
            acc = cm.get("accuracy", 0.0)
            f1m = cm.get("f1_macro", 0.0)
            em = data.get("exact_match", 0.0)
            tf1 = data.get("token_f1", 0.0)
            judge_data = data.get("llm_judge", {})
            judge_avg = judge_data.get("averages", {}).get("nota_geral", 0.0)
            samples = data.get("samples_evaluated", 0)

            table.add_row(
                model,
                f"{acc:.4f}",
                f"{f1m:.4f}",
                f"{em:.4f}",
                f"{tf1:.4f}",
                f"{judge_avg:.2f}" if judge_avg > 0 else "—",
                str(samples),
            )
        except Exception as e:
            table.add_row(f.stem, "erro", "", "", "", "", "")

    console.print(table)


def _display_benchmark_result(result) -> None:
    """Exibe resultados de benchmark formatados."""
    table = Table(title="Métricas de Avaliação")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right", style="green")

    if result.classification_metrics:
        cm = result.classification_metrics
        table.add_row("Accuracy", f"{cm.accuracy:.4f}")
        table.add_row("F1 (macro)", f"{cm.f1_macro:.4f}")
        table.add_row("F1 (weighted)", f"{cm.f1_weighted:.4f}")
        table.add_row("Precision (macro)", f"{cm.precision_macro:.4f}")
        table.add_row("Recall (macro)", f"{cm.recall_macro:.4f}")

    table.add_row("Exact Match", f"{result.exact_match:.4f}")
    table.add_row("Token F1", f"{result.token_f1:.4f}")
    table.add_row("Amostras", str(result.samples_evaluated))
    table.add_row("Tempo (s)", f"{result.inference_time_seconds:.1f}")

    console.print(table)
