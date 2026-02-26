"""
CLI Principal — Interface de linha de comando do MedAssist.

Comandos:
    medassist chat       — Chat interativo com o assistente
    medassist flow       — Fluxo clínico completo para um paciente
    medassist ask        — Pergunta única ao assistente
    medassist train      — Treinamento (fine-tuning, pré-processamento)
    medassist evaluate   — Avaliação (benchmark, LLM-as-judge)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(
    name="medassist",
    help="🏥 MedAssist — Assistente Médico Virtual com IA",
    no_args_is_help=True,
)
console = Console()

# Sub-apps importados depois
from medical_assistant.interfaces.cli.train_cli import train_app
from medical_assistant.interfaces.cli.evaluate_cli import evaluate_app

app.add_typer(train_app, name="train", help="Treinamento e pré-processamento")
app.add_typer(evaluate_app, name="evaluate", help="Avaliação e benchmark")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_services():
    """Carrega serviços (LLM, retriever) com configuração padrão."""
    from medical_assistant.infrastructure.llm.falcon_model import FalconModelAdapter
    from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore
    from medical_assistant.infrastructure.langchain.retrievers import MedicalRetriever

    model_dir = os.getenv("OUTPUT_MODEL_DIR", "models/falcon-7b-qlora")
    base_model = os.getenv("BASE_MODEL_NAME", "tiiuae/falcon-7b-instruct")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    console.print("[dim]Carregando modelo...[/dim]")
    llm_service = FalconModelAdapter(
        model_name=base_model,
        adapter_path=model_dir if Path(model_dir).exists() else None,
    )
    llm_service.load()

    console.print("[dim]Carregando base de conhecimento...[/dim]")
    vector_store = MedicalVectorStore(persist_directory=chroma_dir)
    retriever_service = MedicalRetriever(vector_store=vector_store)

    return llm_service, retriever_service


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pergunta clínica"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Logs detalhados"),
) -> None:
    """Faz uma pergunta única ao assistente médico."""
    _setup_logging(verbose)

    try:
        llm_service, retriever_service = _load_services()

        from medical_assistant.application.use_cases.ask_clinical_question import AskClinicalQuestion
        from medical_assistant.application.dtos.patient_dto import QuestionDTO

        use_case = AskClinicalQuestion(
            llm_service=llm_service,
            retriever_service=retriever_service,
        )

        dto = QuestionDTO(question=question)
        result = use_case.execute(dto)

        console.print(Panel(
            result.answer,
            title="[bold blue]Resposta[/bold blue]",
            border_style="blue",
        ))

        if result.sources:
            console.print("\n[bold]Fontes:[/bold]")
            for src in result.sources:
                console.print(f"  • {src}")

        if result.disclaimer:
            console.print(f"\n[dim italic]{result.disclaimer}[/dim italic]")

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Chat interativo com o assistente médico."""
    _setup_logging(verbose)

    console.print(Panel(
        "🏥 [bold]MedAssist — Chat Interativo[/bold]\n"
        "Digite sua pergunta médica. Comandos especiais:\n"
        "  /sair  — Encerrar sessão\n"
        "  /limpar — Limpar histórico\n"
        "  /ajuda — Mostrar ajuda",
        border_style="green",
    ))

    try:
        llm_service, retriever_service = _load_services()

        from medical_assistant.application.use_cases.ask_clinical_question import AskClinicalQuestion
        from medical_assistant.application.dtos.patient_dto import QuestionDTO

        use_case = AskClinicalQuestion(
            llm_service=llm_service,
            retriever_service=retriever_service,
        )

        console.print("[green]✓ Sistema carregado. Pronto para perguntas.[/green]\n")

        while True:
            try:
                user_input = console.input("[bold cyan]Você > [/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("/sair", "/exit", "/quit"):
                console.print("[dim]Encerrando sessão...[/dim]")
                break

            if user_input.lower() == "/limpar":
                console.clear()
                console.print("[dim]Histórico limpo.[/dim]")
                continue

            if user_input.lower() == "/ajuda":
                console.print(
                    "Faça perguntas sobre condições médicas, medicamentos, "
                    "tratamentos, exames, etc.\n"
                    "O assistente usa base de dados PubMedQA/MedQuAD."
                )
                continue

            dto = QuestionDTO(question=user_input)
            result = use_case.execute(dto)

            console.print(f"\n[bold blue]MedAssist >[/bold blue] {result.answer}")

            if result.sources:
                console.print("[dim]Fontes: " + " | ".join(result.sources[:3]) + "[/dim]")

            if result.disclaimer:
                console.print(f"[dim italic]{result.disclaimer}[/dim italic]")

            console.print()

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def flow(
    patient_file: str = typer.Argument(..., help="Caminho para JSON do paciente"),
    question: str = typer.Option("Avaliação clínica geral", "--question", "-q"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Executa fluxo clínico completo (LangGraph) para um paciente."""
    _setup_logging(verbose)

    try:
        # Carregar dados do paciente
        patient_path = Path(patient_file)
        if not patient_path.exists():
            console.print(f"[red]Arquivo não encontrado: {patient_file}[/red]")
            raise typer.Exit(1)

        with open(patient_path, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

        console.print(Panel(
            f"Paciente: [bold]{patient_data.get('nome', 'N/A')}[/bold]\n"
            f"ID: {patient_data.get('id', 'N/A')}\n"
            f"Pergunta: {question}",
            title="[bold green]Fluxo Clínico[/bold green]",
            border_style="green",
        ))

        # Carregar serviços
        llm_service, retriever_service = _load_services()

        from medical_assistant.infrastructure.langgraph.clinical_graph import ClinicalGraph

        graph = ClinicalGraph(llm_service, retriever_service)
        result = graph.run(patient_data, question)

        # Exibir resultados
        _display_flow_result(result)

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        raise typer.Exit(1)


def _display_flow_result(result: dict) -> None:
    """Exibe resultado do fluxo clínico de forma formatada."""

    # Triagem
    triage = result.get("triage_level", "?")
    triage_colors = {
        "critico": "red",
        "urgente": "yellow",
        "regular": "green",
    }
    color = triage_colors.get(triage, "white")
    console.print(f"\n[bold]Triagem:[/bold] [{color}]{triage.upper()}[/{color}]")
    console.print(f"  {result.get('triage_justification', '')}")

    # Exames
    pending = result.get("pending_exams", [])
    suggested = result.get("suggested_exams", [])
    if pending or suggested:
        console.print("\n[bold]Exames:[/bold]")
        if pending:
            console.print(f"  Pendentes: {', '.join(pending)}")
        if suggested:
            console.print(f"  Sugeridos: {', '.join(suggested)}")

    # Tratamento
    treatment = result.get("treatment_suggestion", "")
    if treatment:
        console.print(Panel(
            treatment,
            title="[bold]Sugestão de Tratamento[/bold]",
            border_style="blue",
        ))

    # Alertas
    alerts = result.get("alerts", [])
    if alerts:
        console.print(f"\n[bold red]⚠ Alertas ({len(alerts)}):[/bold red]")
        for alert in alerts:
            severity = alert.get("severity", "info")
            icon = {"critica": "🔴", "alta": "🟠", "media": "🟡", "baixa": "🟢"}.get(severity, "ℹ")
            console.print(f"  {icon} [{severity.upper()}] {alert.get('message', '')}")

    # Validação
    if result.get("human_validation_required"):
        console.print(Panel(
            f"[bold yellow]VALIDAÇÃO HUMANA NECESSÁRIA[/bold yellow]\n"
            f"Motivo: {result.get('validation_reason', '')}",
            border_style="yellow",
        ))

    # Confiança
    confidence = result.get("confidence_score", 0.0)
    conf_color = "green" if confidence >= 0.7 else ("yellow" if confidence >= 0.4 else "red")
    console.print(f"\n[bold]Confiança:[/bold] [{conf_color}]{confidence:.0%}[/{conf_color}]")

    # Erro
    if result.get("error"):
        console.print(f"\n[red]Erro no fluxo: {result['error']}[/red]")


@app.command()
def patients(
    data_dir: str = typer.Option("data/patients", help="Diretório com dados de pacientes"),
) -> None:
    """Lista pacientes disponíveis para fluxo clínico."""
    patient_dir = Path(data_dir)

    if not patient_dir.exists():
        console.print(f"[yellow]Diretório não encontrado: {data_dir}[/yellow]")
        console.print("[dim]Use 'medassist train generate-patients' para gerar pacientes sintéticos.[/dim]")
        return

    files = list(patient_dir.glob("*.json"))
    if not files:
        console.print("[yellow]Nenhum paciente encontrado.[/yellow]")
        return

    table = Table(title="Pacientes Disponíveis")
    table.add_column("Arquivo", style="cyan")
    table.add_column("ID", style="green")
    table.add_column("Nome", style="white")

    for f in sorted(files):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                table.add_row(f.name, data.get("id", "?"), data.get("nome", "?"))
        except Exception:
            table.add_row(f.name, "?", "[red]erro ao ler[/red]")

    console.print(table)


@app.command()
def version() -> None:
    """Exibe versão do MedAssist."""
    console.print("[bold]MedAssist[/bold] v0.1.0")
    console.print("Assistente Médico Virtual com IA — Projeto Tech Challenge Fase 3")


if __name__ == "__main__":
    app()
