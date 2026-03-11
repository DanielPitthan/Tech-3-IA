"""
Adaptadores para execução de use cases na interface Streamlit.

Centraliza chamadas a use cases e funções de app.py,
fornecendo uma API simples e tipada para a UI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from medical_assistant.application.dtos.patient_dto import (
    ClinicalFlowResultDTO,
    PatientDTO,
    QuestionDTO,
    ResponseDTO,
)
from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.application.use_cases.ask_clinical_question import AskClinicalQuestion
from medical_assistant.application.use_cases.process_patient import ProcessPatient

logger = logging.getLogger(__name__)


# ==============================================================================
# Adaptadores de Casos de Uso
# ==============================================================================


class AdaptadorPerguntaClinica:
    """Adaptador para use case: fazer pergunta clínica."""

    def __init__(
        self,
        llm_service: LLMService,
        retriever_service: RetrieverService | None = None,
    ):
        self.use_case = AskClinicalQuestion(
            llm_service=llm_service,
            retriever=retriever_service,
        )

    def executar(
        self,
        pergunta: str,
        paciente_id: str | None = None,
        contexto_paciente: str = "",
        prioridade: str = "regular",
    ) -> dict[str, Any]:
        """
        Executa pergunta clínica.

        Args:
            pergunta: Texto da pergunta médica
            paciente_id: ID do paciente (opcional)
            contexto_paciente: Contexto adicional do paciente
            prioridade: Prioridade da pergunta

        Returns:
            Dict com resposta estruturada ou erro
        """
        try:
            dto_entrada = QuestionDTO(
                question=pergunta,
                patient_id=paciente_id,
                patient_context=contexto_paciente,
                priority=prioridade,
            )

            resultado: ResponseDTO = self.use_case.execute(dto_entrada)

            return {
                "sucesso": True,
                "resposta": resultado.response_text,
                "fontes": resultado.sources,
                "confianca": resultado.confidence,
                "confianca_label": resultado.confidence_label,
                "disclaimer": resultado.disclaimer,
                "guardrails": resultado.guardrails_triggered,
                "modelo": resultado.model_name,
                "timestamp": resultado.timestamp.isoformat(),
            }

        except Exception as e:
            logger.error("Erro ao executar pergunta clínica: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


class AdaptadorFluxoClinico:
    """Adaptador para use case: fluxo clínico completo."""

    def __init__(
        self,
        llm_service: LLMService,
        retriever_service: RetrieverService | None = None,
    ):
        self.llm_service = llm_service
        self.retriever_service = retriever_service
        self.use_case = ProcessPatient(
            llm_service=llm_service,
            retriever=retriever_service,
        )

    def executar(self, paciente_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Executa fluxo clínico completo para um paciente.

        Args:
            paciente_dict: Dicionário com dados do paciente

        Returns:
            Dict com resultado do fluxo ou erro
        """
        try:
            # Converter dict para PatientDTO
            paciente_dto = PatientDTO(
                id=paciente_dict.get("id", ""),
                nome_anonimizado=paciente_dict.get("nome_anonimizado", "Paciente"),
                idade=paciente_dict.get("idade", 0),
                sexo=paciente_dict.get("sexo", ""),
                setor=paciente_dict.get("setor", ""),
                diagnosticos=paciente_dict.get("diagnosticos", []),
                medicamentos_em_uso=paciente_dict.get("medicamentos_em_uso", []),
                exames=paciente_dict.get("exames", []),
                alergias=paciente_dict.get("alergias", []),
                sinais_vitais=paciente_dict.get("sinais_vitais", {}),
                queixa_principal=paciente_dict.get("queixa_principal", ""),
            )

            resultado: ClinicalFlowResultDTO = self.use_case.execute(paciente_dto)

            return {
                "sucesso": True,
                "paciente_id": resultado.patient_id,
                "triagem_nivel": resultado.triage_level,
                "triagem_descricao": resultado.triage_description,
                "exames_pendentes": resultado.pending_exams,
                "exames_sugeridos": resultado.suggested_exams,
                "tratamento": resultado.treatment_suggestions,
                "alertas": resultado.alerts,
                "validacao_humana_necessaria": resultado.requires_human_validation,
                "motivo_validacao": resultado.validation_reason,
                "log_fluxo": resultado.flow_log,
            }

        except Exception as e:
            logger.error("Erro ao executar fluxo clínico: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


# ==============================================================================
# Adaptadores de Operações de Dados
# ==============================================================================


class AdaptadorPreprocessamento:
    """Adaptador para pré-processamento de dados."""

    @staticmethod
    def executar(
        pubmedqa_file: str,
        ground_truth_file: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Executa pré-processamento de PubMedQA.

        Args:
            pubmedqa_file: Caminho para ori_pqal.json
            ground_truth_file: Caminho para test_ground_truth.json
            output_dir: Diretório de saída

        Returns:
            Dict com resultado ou erro
        """
        try:
            from medical_assistant.data.preprocessing.pubmedqa_processor import (
                PubMedQAProcessor,
            )
            from medical_assistant.data.preprocessing.dataset_splitter import DatasetSplitter

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 1. Processar PubMedQA
            logger.info("Processando PubMedQA...")
            processor = PubMedQAProcessor()
            entries = processor.load_and_process(pubmedqa_file)
            pubmedqa_output = str(output_path / "pubmedqa_processed.jsonl")
            processor.save_jsonl(entries, pubmedqa_output)

            # 2. Gerar arquivo RAG
            logger.info("Gerando arquivo RAG...")
            rag_output = output_path / "medquad_rag.jsonl"
            rag_docs = []

            with open(pubmedqa_output, "r", encoding="utf-8") as f:
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

            # 3. Split do dataset
            logger.info("Dividindo datasets...")
            splitter = DatasetSplitter()
            splits = splitter.split(pubmedqa_output, ground_truth_file, str(output_path))

            return {
                "sucesso": True,
                "pubmedqa_entries": len(entries),
                "rag_documents": len(rag_docs),
                "train_samples": splits.get("train", 0),
                "val_samples": splits.get("val", 0),
                "test_samples": splits.get("test", 0),
                "output_dir": str(output_path),
            }

        except Exception as e:
            logger.error("Erro no pré-processamento: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


class AdaptadorIndexacao:
    """Adaptador para indexação de conhecimento no ChromaDB."""

    @staticmethod
    def executar(
        rag_file: str,
        chroma_dir: str,
    ) -> dict[str, Any]:
        """
        Indexa documentos RAG no ChromaDB.

        Args:
            rag_file: Caminho para arquivo RAG JSONL
            chroma_dir: Diretório para persistência ChromaDB

        Returns:
            Dict com resultado ou erro
        """
        try:
            from medical_assistant.infrastructure.persistence.vector_store import (
                MedicalVectorStore,
            )

            logger.info("Carregando documentos RAG...")
            documents: list[dict] = []

            with open(rag_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    documents.append({
                        "content": entry.get("content", ""),
                        "metadata": entry.get("metadata", {}),
                    })

            logger.info("Indexando %d documentos no ChromaDB...", len(documents))
            store = MedicalVectorStore(persist_directory=chroma_dir)
            store.add_documents(documents)

            return {
                "sucesso": True,
                "documentos_indexados": len(documents),
                "chroma_dir": str(chroma_dir),
            }

        except Exception as e:
            logger.error("Erro na indexação: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


class AdaptadorGeracaoPacientes:
    """Adaptador para geração de pacientes sintéticos."""

    @staticmethod
    def executar(count: int, output_dir: str) -> dict[str, Any]:
        """
        Gera pacientes sintéticos.

        Args:
            count: Número de pacientes a gerar
            output_dir: Diretório de saída

        Returns:
            Dict com resultado ou erro
        """
        try:
            from medical_assistant.data.synthetic.synthetic_patients import (
                SyntheticPatientGenerator,
            )

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info("Gerando %d pacientes sintéticos...", count)
            generator = SyntheticPatientGenerator()
            patients = generator.generate_batch(count)

            for patient in patients:
                filepath = output_path / f"patient_{patient['id']}.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(patient, f, indent=2, ensure_ascii=False)

            return {
                "sucesso": True,
                "pacientes_gerados": len(patients),
                "output_dir": str(output_path),
            }

        except Exception as e:
            logger.error("Erro na geração de pacientes: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


# ==============================================================================
# Adaptadores de Avaliação
# ==============================================================================


class AdaptadorBenchmark:
    """Adaptador para avaliação quantitativa (benchmark)."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def executar(
        self,
        test_file: str,
        max_samples: int | None = None,
        output_dir: str = "evaluation_results",
    ) -> dict[str, Any]:
        """
        Executa benchmark quantitativo.

        Args:
            test_file: Arquivo de teste JSONL
            max_samples: Limite de amostras (None = todas)
            output_dir: Diretório de saída

        Returns:
            Dict com métricas ou erro
        """
        try:
            from medical_assistant.evaluation.benchmark import BenchmarkRunner

            logger.info("Iniciando benchmark...")
            runner = BenchmarkRunner(llm_service=self.llm_service, output_dir=output_dir)
            result = runner.run(
                test_file=test_file,
                model_name="ollama-llama3",
                max_samples=max_samples,
            )

            metrics = {
                "sucesso": True,
                "amostras_avaliadas": result.samples_evaluated,
                "accuracy": float(result.classification_metrics.accuracy) if result.classification_metrics else 0.0,
                "f1_macro": float(result.classification_metrics.f1_macro) if result.classification_metrics else 0.0,
                "f1_weighted": float(result.classification_metrics.f1_weighted) if result.classification_metrics else 0.0,
                "exact_match": float(result.exact_match),
                "token_f1": float(result.token_f1),
                "tempo_inference_s": float(result.inference_time_seconds),
            }

            return metrics

        except Exception as e:
            logger.error("Erro no benchmark: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


class AdaptadorAvaliadorLLM:
    """Adaptador para avaliação qualitativa (LLM-as-judge)."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def executar(
        self,
        test_file: str,
        judge_model: str = "gpt-4o-mini",
        max_samples: int = 50,
        output_dir: str = "evaluation_results",
    ) -> dict[str, Any]:
        """
        Executa avaliação qualitativa com LLM-as-judge.

        Args:
            test_file: Arquivo de teste JSONL
            judge_model: Modelo para juíz (ex: gpt-4o-mini)
            max_samples: Limit de amostras
            output_dir: Diretório de saída

        Returns:
            Dict com resultado ou erro
        """
        try:
            from medical_assistant.evaluation.benchmark import BenchmarkRunner
            from medical_assistant.evaluation.llm_judge import LLMJudge

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")

            logger.info("Iniciando avaliação com LLM-as-judge...")
            llm_judge = LLMJudge(model=judge_model, api_key=api_key)
            runner = BenchmarkRunner(
                llm_service=self.llm_service,
                judge=llm_judge,
                output_dir=output_dir,
            )

            result = runner.run(
                test_file=test_file,
                model_name="ollama-llama3",
                max_samples=max_samples,
                run_judge=True,
                judge_max_samples=max_samples,
            )

            metrics = {
                "sucesso": True,
                "amostras_avaliadas": result.samples_evaluated,
                "accuracy": float(result.classification_metrics.accuracy) if result.classification_metrics else 0.0,
                "exact_match": float(result.exact_match),
                "token_f1": float(result.token_f1),
            }

            if result.judge_report:
                judge_summary = result.judge_report.summary()
                metrics["judge_summary"] = judge_summary

            return metrics

        except Exception as e:
            logger.error("Erro na avaliação com judge: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }


class AdaptadorRelatorioAvaliacao:
    """Adaptador para geração de relatório consolidado de avaliação."""

    @staticmethod
    def executar(eval_output_dir: str) -> dict[str, Any]:
        """
        Gera relatório consolidado de resultados de avaliação.

        Args:
            eval_output_dir: Diretório com resultados de benchmark

        Returns:
            Dict com relatório ou erro
        """
        try:
            results_path = Path(eval_output_dir)

            if not results_path.exists():
                return {
                    "sucesso": False,
                    "erro": f"Diretório não encontrado: {eval_output_dir}",
                }

            json_files = sorted(results_path.glob("benchmark_*.json"))

            if not json_files:
                return {
                    "sucesso": True,
                    "resultados": [],
                    "mensagem": "Nenhum resultado de benchmark encontrado",
                }

            resultados = []

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

                    resultados.append({
                        "arquivo": f.name,
                        "modelo": model,
                        "accuracy": float(acc),
                        "f1_macro": float(f1m),
                        "exact_match": float(em),
                        "token_f1": float(tf1),
                        "judge_score": float(judge_avg) if judge_avg > 0 else None,
                        "amostras": samples,
                    })
                except Exception as e:
                    logger.warning("Erro ao ler %s: %s", f.name, e)

            return {
                "sucesso": True,
                "resultados": resultados,
                "total_benchmarks": len(resultados),
            }

        except Exception as e:
            logger.error("Erro ao gerar relatório: %s", e, exc_info=True)
            return {
                "sucesso": False,
                "erro": str(e),
                "tipo_erro": type(e).__name__,
            }
