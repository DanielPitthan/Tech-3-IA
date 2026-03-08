#!/usr/bin/env python3
"""
MedAssist — Ponto de entrada principal via linha de comando.

Uso:
    python app.py --preprocess
    python app.py --finetune [--config FILE] [-v]
    python app.py --index-knowledge
    python app.py --generate-patients [--count N]
    python app.py --chat
    python app.py --ask "Pergunta clínica"
    python app.py --flow PATIENT.json [-q "Pergunta"]
    python app.py --evaluate benchmark|judge|report [--max-samples N]
    python app.py --patients
    python app.py --all
    python app.py --runall
    python app.py --version
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_services():
    """Carrega serviços (LLM, retriever) com configuração padrão."""
    from medical_assistant.infrastructure.llm.ollama_model import OllamaModelAdapter
    from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore
    from medical_assistant.infrastructure.langchain.retrievers import MedicalRetriever

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    console.print("[dim]Carregando modelo Ollama (Llama 3)...[/dim]")
    llm_service = OllamaModelAdapter(
        model_name=ollama_model,
        base_url=ollama_url,
    )
    llm_service.load()

    console.print("[dim]Carregando base de conhecimento...[/dim]")
    vector_store = MedicalVectorStore(persist_directory=chroma_dir)
    retriever_service = MedicalRetriever(vector_store=vector_store)

    return llm_service, retriever_service


def _display_flow_result(result: dict) -> None:
    """Exibe resultado do fluxo clínico de forma formatada."""
    triage = result.get("triage_level", "?")
    triage_colors = {"critico": "red", "urgente": "yellow", "regular": "green"}
    color = triage_colors.get(triage, "white")
    console.print(f"\n[bold]Triagem:[/bold] [{color}]{triage.upper()}[/{color}]")
    console.print(f"  {result.get('triage_justification', '')}")

    pending = result.get("pending_exams", [])
    suggested = result.get("suggested_exams", [])
    if pending or suggested:
        console.print("\n[bold]Exames:[/bold]")
        if pending:
            console.print(f"  Pendentes: {', '.join(pending)}")
        if suggested:
            console.print(f"  Sugeridos: {', '.join(suggested)}")

    treatment = result.get("treatment_suggestion", "")
    if treatment:
        console.print(Panel(
            treatment,
            title="[bold]Sugestão de Tratamento[/bold]",
            border_style="blue",
        ))

    alerts = result.get("alerts", [])
    if alerts:
        console.print(f"\n[bold red]⚠ Alertas ({len(alerts)}):[/bold red]")
        for alert in alerts:
            severity = alert.get("severity", "info")
            icon = {"critica": "🔴", "alta": "🟠", "media": "🟡", "baixa": "🟢"}.get(severity, "ℹ")
            console.print(f"  {icon} [{severity.upper()}] {alert.get('message', '')}")

    if result.get("human_validation_required"):
        console.print(Panel(
            f"[bold yellow]VALIDAÇÃO HUMANA NECESSÁRIA[/bold yellow]\n"
            f"Motivo: {result.get('validation_reason', '')}",
            border_style="yellow",
        ))

    confidence = result.get("confidence_score", 0.0)
    conf_color = "green" if confidence >= 0.7 else ("yellow" if confidence >= 0.4 else "red")
    console.print(f"\n[bold]Confiança:[/bold] [{conf_color}]{confidence:.0%}[/{conf_color}]")

    if result.get("error"):
        console.print(f"\n[red]Erro no fluxo: {result['error']}[/red]")


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


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_preprocess(args) -> None:
    """Pré-processar PubMedQA para treinamento e gerar artefatos derivados."""
    _setup_logging(args.verbose)
    output_path = Path(args.output)
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
            entries = processor.load_and_process(args.pubmedqa)
            pubmed_output = str(output_path / "pubmedqa_processed.jsonl")
            processor.save_jsonl(entries, pubmed_output)
            progress.update(task1, description=f"✓ PubMedQA: {len(entries)} entradas")
        except Exception as e:
            progress.update(task1, description=f"✗ PubMedQA: {e}")
            console.print(f"[red]Erro PubMedQA: {e}[/red]")

        # 2. Gerar arquivo RAG compatível com o fluxo atual
        task2 = progress.add_task("Gerando base RAG (compatibilidade)...", total=None)
        try:
            pubmed_output = output_path / "pubmedqa_processed.jsonl"
            if not pubmed_output.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {pubmed_output}")

            rag_output = output_path / "medquad_rag.jsonl"
            rag_docs = []

            with open(pubmed_output, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    content = f"{entry.get('input', '')}\n\n{entry.get('output', '')}".strip()
                    rag_docs.append({
                        "content": content,
                        "metadata": {
                            "source": "pubmedqa",
                            "pmid": entry.get("pmid", ""),
                            "label": entry.get("label", ""),
                        },
                    })

            with open(rag_output, "w", encoding="utf-8") as f:
                for doc in rag_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            progress.update(task2, description=f"✓ RAG: {len(rag_docs)} documentos")
        except Exception as e:
            progress.update(task2, description=f"✗ RAG: {e}")
            console.print(f"[red]Erro geração RAG: {e}[/red]")

        # 3. Split do dataset
        task3 = progress.add_task("Dividindo datasets...", total=None)
        try:
            from medical_assistant.data.preprocessing.dataset_splitter import DatasetSplitter

            splitter = DatasetSplitter()
            combined = str(output_path / "pubmedqa_processed.jsonl")
            splits = splitter.split(combined, args.ground_truth, str(output_path))
            progress.update(
                task3,
                description=f"✓ Split: train={splits.get('train', 0)}, "
                            f"val={splits.get('val', 0)}, test={splits.get('test', 0)}",
            )
        except Exception as e:
            progress.update(task3, description=f"✗ Split: {e}")
            console.print(f"[red]Erro Split: {e}[/red]")

    console.print("\n[green]✓ Pré-processamento concluído![/green]")
    console.print(f"Dados em: [cyan]{args.output}[/cyan]")


def cmd_finetune(args) -> None:
    """Executar fine-tuning QLoRA do Llama 3.1-8B."""
    _setup_logging(args.verbose)

    console.print(Panel(
        f"[bold]Fine-tuning QLoRA[/bold]\n"
        f"Config: {args.config}\n"
        f"Modelo: {args.model_config}\n"
        f"Treino: {args.train_file}\n"
        f"Saída: {args.output}",
        title="🔧 Fine-tuning",
        border_style="blue",
    ))

    try:
        from medical_assistant.infrastructure.llm.model_config import ModelConfig
        from medical_assistant.infrastructure.llm.llama3_qlora_trainer import Llama3QLoRATrainer

        mc = ModelConfig.from_yaml(
            model_config_path=args.model_config,
            qlora_config_path=args.config,
        )
        mc.training.output_dir = args.output

        trainer = Llama3QLoRATrainer(config=mc)

        console.print("[dim]Iniciando pipeline de fine-tuning...[/dim]")
        trainer.run_pipeline(args.train_file, args.val_file)

        console.print(f"\n[green]✓ Fine-tuning concluído! Modelo salvo em: {args.output}[/green]")

    except Exception as e:
        console.print(f"[red]Erro no fine-tuning: {e}[/red]")
        sys.exit(1)


def cmd_index_knowledge(args) -> None:
    """Indexar base de conhecimento no ChromaDB para RAG."""
    _setup_logging(args.verbose)

    try:
        from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore

        documents: list[dict] = []

        with open(args.rag_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                documents.append({
                    "content": entry.get("content", ""),
                    "metadata": entry.get("metadata", {}),
                })

        console.print(f"[dim]Carregados {len(documents)} documentos[/dim]")

        store = MedicalVectorStore(persist_directory=args.chroma_dir)
        store.add_documents(documents)

        console.print(f"[green]✓ {len(documents)} documentos indexados no ChromaDB[/green]")
        console.print(f"[dim]Persistent directory: {args.chroma_dir}[/dim]")

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


def cmd_generate_patients(args) -> None:
    """Gerar pacientes sintéticos para testes."""
    _setup_logging(args.verbose)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from medical_assistant.data.synthetic.synthetic_patients import SyntheticPatientGenerator

        generator = SyntheticPatientGenerator()
        patients = generator.generate_batch(args.count)

        for patient in patients:
            filepath = output_path / f"patient_{patient['id']}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(patient, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✓ {len(patients)} pacientes gerados em {args.output}[/green]")

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


def cmd_chat(args) -> None:
    """Chat interativo com o assistente médico."""
    _setup_logging(args.verbose)

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
        sys.exit(1)


def cmd_ask(args) -> None:
    """Faz uma pergunta única ao assistente médico."""
    _setup_logging(args.verbose)

    try:
        llm_service, retriever_service = _load_services()

        from medical_assistant.application.use_cases.ask_clinical_question import AskClinicalQuestion
        from medical_assistant.application.dtos.patient_dto import QuestionDTO

        use_case = AskClinicalQuestion(
            llm_service=llm_service,
            retriever_service=retriever_service,
        )

        dto = QuestionDTO(question=args.ask)
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
        sys.exit(1)


def cmd_flow(args) -> None:
    """Executa fluxo clínico completo (LangGraph) para um paciente."""
    _setup_logging(args.verbose)

    try:
        patient_path = Path(args.flow)
        if not patient_path.exists():
            console.print(f"[red]Arquivo não encontrado: {args.flow}[/red]")
            sys.exit(1)

        with open(patient_path, "r", encoding="utf-8") as f:
            patient_data = json.load(f)

        console.print(Panel(
            f"Paciente: [bold]{patient_data.get('nome', 'N/A')}[/bold]\n"
            f"ID: {patient_data.get('id', 'N/A')}\n"
            f"Pergunta: {args.question}",
            title="[bold green]Fluxo Clínico[/bold green]",
            border_style="green",
        ))

        llm_service, retriever_service = _load_services()

        from medical_assistant.infrastructure.langgraph.clinical_graph import ClinicalGraph

        graph = ClinicalGraph(llm_service, retriever_service)
        result = graph.run(patient_data, args.question)

        _display_flow_result(result)

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


def cmd_evaluate(args) -> None:
    """Executa avaliação: benchmark, judge ou report."""
    _setup_logging(args.verbose)

    mode = args.evaluate
    if mode == "benchmark":
        _run_benchmark(args)
    elif mode == "judge":
        _run_judge(args)
    elif mode == "report":
        _run_report(args)


def _run_benchmark(args) -> None:
    """Executar benchmark de métricas quantitativas."""
    console.print(Panel(
        f"[bold]Benchmark Quantitativo[/bold]\n"
        f"Dataset: {args.test_file}\n"
        f"Modelo: Ollama/Llama 3\n"
        f"Amostras: {args.max_samples or 'todas'}",
        title="📊 Benchmark",
        border_style="blue",
    ))

    try:
        from medical_assistant.infrastructure.llm.ollama_model import OllamaModelAdapter
        from medical_assistant.evaluation.benchmark import BenchmarkRunner

        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        console.print("[dim]Carregando modelo Ollama (Llama 3)...[/dim]")
        llm_service = OllamaModelAdapter(
            model_name=ollama_model,
            base_url=ollama_url,
        )
        llm_service.load()

        runner = BenchmarkRunner(llm_service=llm_service, output_dir=args.eval_output)
        result = runner.run(
            test_file=args.test_file,
            model_name="ollama-llama3",
            max_samples=args.max_samples,
        )

        _display_benchmark_result(result)

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


def _run_judge(args) -> None:
    """Avaliação qualitativa com LLM-as-judge."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        console.print("[red]OPENAI_API_KEY não configurada. Defina no .env[/red]")
        sys.exit(1)

    max_samples = args.max_samples or 50

    console.print(Panel(
        f"[bold]LLM-as-Judge[/bold]\n"
        f"Judge: {args.judge_model}\n"
        f"Amostras: {max_samples}",
        title="🧑‍⚖️ Avaliação Qualitativa",
        border_style="magenta",
    ))

    try:
        from medical_assistant.infrastructure.llm.ollama_model import OllamaModelAdapter
        from medical_assistant.evaluation.benchmark import BenchmarkRunner
        from medical_assistant.evaluation.llm_judge import LLMJudge

        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        console.print("[dim]Carregando modelo Ollama (Llama 3)...[/dim]")
        llm_service = OllamaModelAdapter(
            model_name=ollama_model,
            base_url=ollama_url,
        )
        llm_service.load()

        llm_judge = LLMJudge(model=args.judge_model, api_key=api_key)

        runner = BenchmarkRunner(
            llm_service=llm_service,
            judge=llm_judge,
            output_dir=args.eval_output,
        )

        result = runner.run(
            test_file=args.test_file,
            model_name="ollama-llama3",
            max_samples=max_samples,
            run_judge=True,
            judge_max_samples=max_samples,
        )

        _display_benchmark_result(result)

        if result.judge_report:
            console.print(result.judge_report.summary())

    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


def _run_report(args) -> None:
    """Gerar relatório consolidado de avaliação."""
    results_path = Path(args.eval_output)

    if not results_path.exists():
        console.print(f"[red]Diretório não encontrado: {args.eval_output}[/red]")
        sys.exit(1)

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
        except Exception:
            table.add_row(f.stem, "erro", "", "", "", "", "")

    console.print(table)


def cmd_patients(args) -> None:
    """Lista pacientes disponíveis para fluxo clínico."""
    patient_dir = Path(args.patients_dir)

    if not patient_dir.exists():
        console.print(f"[yellow]Diretório não encontrado: {args.patients_dir}[/yellow]")
        console.print("[dim]Use 'python app.py --generate-patients' para gerar pacientes sintéticos.[/dim]")
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


def cmd_version(_args) -> None:
    """Exibe versão do MedAssist."""
    console.print("[bold]MedAssist[/bold] v1.0.0")
    console.print("Assistente Médico Virtual com IA — Projeto Tech Challenge Fase 3")


def cmd_all(args) -> None:
    """Executa o pipeline completo de forma interativa."""
    from rich.prompt import Confirm, Prompt

    _setup_logging(args.verbose)

    console.print(Panel(
        "🏥 [bold]MedAssist — Pipeline Completo[/bold]\n"
        "Este modo executa todas as etapas em sequência.\n"
        "Você será solicitado a confirmar cada etapa.",
        border_style="green",
    ))

    # 1. Preprocess
    if Confirm.ask("\n[bold]1/7[/bold] Executar pré-processamento de dados?", default=True):
        args.output = Prompt.ask("  Diretório de saída", default="data/processed")
        args.pubmedqa = Prompt.ask("  Arquivo PubMedQA", default="DataSets/ori_pqal.json")
        args.ground_truth = Prompt.ask("  Ground truth", default="DataSets/test_ground_truth.json")
        cmd_preprocess(args)
        console.print()

    # 2. Index knowledge
    if Confirm.ask("[bold]2/7[/bold] Indexar base de conhecimento no ChromaDB?", default=True):
        args.rag_file = Prompt.ask("  Arquivo RAG", default="data/processed/medquad_rag.jsonl")
        args.chroma_dir = Prompt.ask("  Diretório ChromaDB", default="chroma_db")
        cmd_index_knowledge(args)
        console.print()

    # 3. Fine-tuning
    if Confirm.ask("[bold]3/7[/bold] Executar fine-tuning QLoRA?", default=True):
        args.config = Prompt.ask("  Config LoRA", default="configs/qlora_config.yaml")
        args.model_config = Prompt.ask("  Config modelo", default="configs/model_config.yaml")
        args.train_file = Prompt.ask("  Arquivo treino", default="data/processed/dataset_train.jsonl")
        args.val_file = Prompt.ask("  Arquivo validação", default="data/processed/dataset_val.jsonl")
        args.output = Prompt.ask("  Diretório de saída", default="models/llama3-qlora")
        cmd_finetune(args)
        console.print()

    # 4. Generate patients
    if Confirm.ask("[bold]4/7[/bold] Gerar pacientes sintéticos?", default=True):
        args.count = int(Prompt.ask("  Quantidade", default="20"))
        args.output = Prompt.ask("  Diretório de saída", default="data/patients")
        cmd_generate_patients(args)
        console.print()

    # 5. Benchmark
    if Confirm.ask("[bold]5/7[/bold] Executar benchmark quantitativo?", default=True):
        args.test_file = Prompt.ask("  Arquivo de teste", default="data/processed/dataset_test.jsonl")
        args.model_dir = Prompt.ask("  Diretório do modelo", default="models/llama3-qlora")
        args.max_samples = int(Prompt.ask("  Máximo de amostras (0=todas)", default="100")) or None
        args.eval_output = Prompt.ask("  Diretório de resultados", default="evaluation_results")
        _run_benchmark(args)
        console.print()

    # 6. Judge
    if Confirm.ask("[bold]6/7[/bold] Executar avaliação LLM-as-Judge?", default=True):
        args.test_file = Prompt.ask("  Arquivo de teste", default="data/processed/dataset_test.jsonl")
        args.model_dir = Prompt.ask("  Diretório do modelo", default="models/llama3-qlora")
        args.judge_model = Prompt.ask("  Modelo judge", default="gpt-4o-mini")
        args.max_samples = int(Prompt.ask("  Máximo de amostras", default="50")) or None
        args.eval_output = Prompt.ask("  Diretório de resultados", default="evaluation_results")
        _run_judge(args)
        console.print()

    # 7. Report
    if Confirm.ask("[bold]7/7[/bold] Gerar relatório consolidado?", default=True):
        args.eval_output = Prompt.ask("  Diretório de resultados", default="evaluation_results")
        _run_report(args)
        console.print()

    console.print(Panel(
        "[bold green]✓ Pipeline completo finalizado![/bold green]",
        border_style="green",
    ))


def cmd_runall(args) -> None:
    """Executa o pipeline completo automaticamente (sem interação e sem testes unitários)."""
    _setup_logging(args.verbose)

    console.print(Panel(
        "🏥 [bold]MedAssist — Pipeline Automático (runall)[/bold]\n"
        "Executando todas as etapas em sequência com valores padrão.\n"
        "Testes unitários NÃO serão executados.",
        border_style="green",
    ))

    steps_ok = 0
    steps_fail = 0

    # 1. Pré-processamento
    console.print("\n[bold cyan]━━━ Etapa 1/6: Pré-processamento ━━━[/bold cyan]")
    try:
        args.output = args.output or "data/processed"
        args.pubmedqa = getattr(args, "pubmedqa", "DataSets/ori_pqal.json")
        args.ground_truth = getattr(args, "ground_truth", "DataSets/test_ground_truth.json")
        cmd_preprocess(args)
        steps_ok += 1
    except Exception as e:
        console.print(f"[red]Erro no pré-processamento: {e}[/red]")
        steps_fail += 1

    # 2. Indexação de conhecimento
    console.print("\n[bold cyan]━━━ Etapa 2/6: Indexação ChromaDB ━━━[/bold cyan]")
    try:
        args.rag_file = getattr(args, "rag_file", "data/processed/medquad_rag.jsonl")
        args.chroma_dir = getattr(args, "chroma_dir", "chroma_db")
        cmd_index_knowledge(args)
        steps_ok += 1
    except Exception as e:
        console.print(f"[red]Erro na indexação: {e}[/red]")
        steps_fail += 1

    # 3. Geração de pacientes sintéticos
    console.print("\n[bold cyan]━━━ Etapa 3/6: Geração de Pacientes ━━━[/bold cyan]")
    try:
        args.count = getattr(args, "count", 20)
        args.output = "data/patients"
        cmd_generate_patients(args)
        steps_ok += 1
    except Exception as e:
        console.print(f"[red]Erro na geração de pacientes: {e}[/red]")
        steps_fail += 1

    # 4. Benchmark quantitativo
    console.print("\n[bold cyan]━━━ Etapa 4/6: Benchmark ━━━[/bold cyan]")
    try:
        args.test_file = getattr(args, "test_file", "data/processed/dataset_test.jsonl")
        args.max_samples = getattr(args, "max_samples", None) or 100
        args.eval_output = getattr(args, "eval_output", "evaluation_results")
        _run_benchmark(args)
        steps_ok += 1
    except Exception as e:
        console.print(f"[red]Erro no benchmark: {e}[/red]")
        steps_fail += 1

    # 5. Avaliação LLM-as-Judge
    console.print("\n[bold cyan]━━━ Etapa 5/6: LLM-as-Judge ━━━[/bold cyan]")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            args.judge_model = getattr(args, "judge_model", "gpt-4o-mini")
            args.max_samples = 50
            _run_judge(args)
            steps_ok += 1
        except Exception as e:
            console.print(f"[red]Erro no LLM-as-Judge: {e}[/red]")
            steps_fail += 1
    else:
        console.print("[yellow]⚠ OPENAI_API_KEY não configurada — pulando LLM-as-Judge.[/yellow]")
        steps_fail += 1

    # 6. Relatório consolidado
    console.print("\n[bold cyan]━━━ Etapa 6/6: Relatório ━━━[/bold cyan]")
    try:
        args.eval_output = getattr(args, "eval_output", "evaluation_results")
        _run_report(args)
        steps_ok += 1
    except Exception as e:
        console.print(f"[red]Erro no relatório: {e}[/red]")
        steps_fail += 1

    # Resumo final
    total = steps_ok + steps_fail
    color = "green" if steps_fail == 0 else "yellow"
    console.print(Panel(
        f"[bold {color}]Pipeline finalizado: {steps_ok}/{total} etapas concluídas com sucesso.[/bold {color}]",
        border_style=color,
    ))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description="🏥 MedAssist — Assistente Médico Virtual com IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python app.py --preprocess\n"
            "  python app.py --finetune --config configs/qlora_config.yaml -v\n"
            "  python app.py --generate-patients --count 20\n"
            "  python app.py --chat\n"
            "  python app.py --ask \"Quais são os efeitos colaterais da metformina?\"\n"
            "  python app.py --flow data/patients/patient_P001.json -q \"Avaliação pré-operatória\"\n"
            "  python app.py --evaluate benchmark --max-samples 100\n"
            "  python app.py --evaluate judge --max-samples 50\n"
            "  python app.py --evaluate report\n"
            "  python app.py --patients\n"
            "  python app.py --all\n"
            "  python app.py --runall\n"
        ),
    )

    # Grupo de ações mutuamente exclusivas
    action = parser.add_mutually_exclusive_group(required=True)

    action.add_argument(
        "--preprocess",
        action="store_true",
        help="Pré-processar PubMedQA para treinamento e gerar splits fixos por ground truth",
    )
    action.add_argument(
        "--finetune",
        action="store_true",
        help="Executar fine-tuning QLoRA do Llama 3.1-8B",
    )
    action.add_argument(
        "--index-knowledge",
        action="store_true",
        help="Indexar base de conhecimento no ChromaDB para RAG",
    )
    action.add_argument(
        "--generate-patients",
        action="store_true",
        help="Gerar pacientes sintéticos para testes",
    )
    action.add_argument(
        "--chat",
        action="store_true",
        help="Chat interativo com o assistente médico",
    )
    action.add_argument(
        "--ask",
        type=str,
        metavar="PERGUNTA",
        help="Pergunta única ao assistente médico",
    )
    action.add_argument(
        "--flow",
        type=str,
        metavar="PACIENTE_JSON",
        help="Executar fluxo clínico completo para um paciente (caminho JSON)",
    )
    action.add_argument(
        "--evaluate",
        type=str,
        choices=["benchmark", "judge", "report"],
        metavar="{benchmark,judge,report}",
        help="Avaliação: benchmark | judge | report",
    )
    action.add_argument(
        "--patients",
        action="store_true",
        help="Listar pacientes disponíveis",
    )
    action.add_argument(
        "--all",
        action="store_true",
        help="Executar pipeline completo (interativo)",
    )
    action.add_argument(
        "--runall",
        action="store_true",
        help="Executar pipeline completo automaticamente (sem interação, sem testes unitários)",
    )
    action.add_argument(
        "--version",
        action="store_true",
        help="Exibir versão do MedAssist",
    )

    # Opções comuns
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Logs detalhados (debug)",
    )

    # Opções de preprocess
    preprocess_group = parser.add_argument_group("Opções de pré-processamento")
    preprocess_group.add_argument(
        "--pubmedqa",
        default="DataSets/ori_pqal.json",
        help="Caminho para ori_pqal.json (PubMedQA) (default: DataSets/ori_pqal.json)",
    )
    preprocess_group.add_argument(
        "--ground-truth",
        default="DataSets/test_ground_truth.json",
        dest="ground_truth",
        help="Caminho para test_ground_truth.json",
    )

    # Opções de finetune
    finetune_group = parser.add_argument_group("Opções de fine-tuning")
    finetune_group.add_argument(
        "--config", "-c",
        default="configs/qlora_config.yaml",
        help="Config QLoRA YAML (default: configs/qlora_config.yaml)",
    )
    finetune_group.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        dest="model_config",
        help="Config do modelo YAML (default: configs/model_config.yaml)",
    )
    finetune_group.add_argument(
        "--train-file",
        default="data/processed/dataset_train.jsonl",
        dest="train_file",
        help="Arquivo de treino JSONL (default: data/processed/dataset_train.jsonl)",
    )
    finetune_group.add_argument(
        "--val-file",
        default="data/processed/dataset_val.jsonl",
        dest="val_file",
        help="Arquivo de validação JSONL (default: data/processed/dataset_val.jsonl)",
    )

    # Opções de output (compartilhado)
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Diretório de saída (contexto varia por comando)",
    )

    # Opções de index-knowledge
    index_group = parser.add_argument_group("Opções de indexação")
    index_group.add_argument(
        "--rag-file",
        default="data/processed/medquad_rag.jsonl",
        dest="rag_file",
        help="Arquivo RAG JSONL (default: data/processed/medquad_rag.jsonl)",
    )
    index_group.add_argument(
        "--chroma-dir",
        default="chroma_db",
        dest="chroma_dir",
        help="Diretório ChromaDB (default: chroma_db)",
    )

    # Opções de generate-patients
    patient_gen_group = parser.add_argument_group("Opções de geração de pacientes")
    patient_gen_group.add_argument(
        "--count", "-n",
        type=int,
        default=20,
        help="Número de pacientes sintéticos (default: 20)",
    )

    # Opções de flow
    flow_group = parser.add_argument_group("Opções de fluxo clínico")
    flow_group.add_argument(
        "-q", "--question",
        default="Avaliação clínica geral",
        help="Pergunta clínica para o fluxo (default: 'Avaliação clínica geral')",
    )

    # Opções de evaluate
    eval_group = parser.add_argument_group("Opções de avaliação")
    eval_group.add_argument(
        "--test-file", "-t",
        default="data/processed/dataset_test.jsonl",
        dest="test_file",
        help="Arquivo de teste JSONL (default: data/processed/dataset_test.jsonl)",
    )
    eval_group.add_argument(
        "--model-dir", "-m",
        default="models/llama3-qlora",
        dest="model_dir",
        help="Diretório do modelo (default: models/llama3-qlora)",
    )
    eval_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        dest="max_samples",
        help="Máximo de amostras para avaliação",
    )
    eval_group.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        dest="judge_model",
        help="Modelo para LLM-as-judge (default: gpt-4o-mini)",
    )
    eval_group.add_argument(
        "--eval-output",
        default="evaluation_results",
        dest="eval_output",
        help="Diretório de resultados de avaliação (default: evaluation_results)",
    )

    # Opções de patients
    patients_group = parser.add_argument_group("Opções de listagem de pacientes")
    patients_group.add_argument(
        "--patients-dir",
        default="data/patients",
        dest="patients_dir",
        help="Diretório com dados de pacientes (default: data/patients)",
    )

    return parser


def _resolve_output(args) -> None:
    """Define defaults para --output quando não informado."""
    if args.output is not None:
        return
    if getattr(args, "preprocess", False):
        args.output = "data/processed"
    elif getattr(args, "finetune", False):
        args.output = "models/llama3-qlora"
    elif getattr(args, "generate_patients", False):
        args.output = "data/patients"
    elif getattr(args, "runall", False):
        args.output = "data/processed"
    else:
        args.output = "."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _resolve_output(args)

    if args.version:
        cmd_version(args)
    elif args.preprocess:
        cmd_preprocess(args)
    elif args.finetune:
        cmd_finetune(args)
    elif args.index_knowledge:
        cmd_index_knowledge(args)
    elif args.generate_patients:
        cmd_generate_patients(args)
    elif args.chat:
        cmd_chat(args)
    elif args.ask:
        cmd_ask(args)
    elif args.flow:
        cmd_flow(args)
    elif args.evaluate:
        cmd_evaluate(args)
    elif args.patients:
        cmd_patients(args)
    elif args.all:
        cmd_all(args)
    elif args.runall:
        cmd_runall(args)


if __name__ == "__main__":
    main()
