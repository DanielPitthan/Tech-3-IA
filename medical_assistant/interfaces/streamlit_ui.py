"""
Interface Streamlit para MedAssist.

Fornece UI profissional para descobrir, configurar e executar
pipelines do assistente médico virtual com IA.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv

from medical_assistant.application.interfaces.llm_service import LLMService
from medical_assistant.application.interfaces.retriever_service import RetrieverService
from medical_assistant.infrastructure.llm.ollama_model import OllamaModelAdapter
from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore
from medical_assistant.infrastructure.langchain.retrievers import MedicalRetriever

from medical_assistant.interfaces.ui_adapter import (
    AdaptadorPerguntaClinica,
    AdaptadorFluxoClinico,
    AdaptadorPreprocessamento,
    AdaptadorIndexacao,
    AdaptadorGeracaoPacientes,
    AdaptadorBenchmark,
    AdaptadorAvaliadorLLM,
    AdaptadorRelatorioAvaliacao,
)

# Carregar variáveis de ambiente
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuração Streamlit
# ==============================================================================

st.set_page_config(
    page_title="🏥 MedAssist — Assistente Médico com IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Cache de Serviços
# ==============================================================================


@st.cache_resource
def _carrega_llm_service() -> LLMService:
    """Carrega e cacheia OllamaModelAdapter."""
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        logger.info("Carregando modelo Ollama...")
        llm_service = OllamaModelAdapter(
            model_name=ollama_model,
            base_url=ollama_url,
        )
        llm_service.load()
        logger.info("Modelo Ollama carregado com sucesso")
        return llm_service
    except Exception as e:
        logger.error("Erro ao carregar LLM: %s", e)
        raise


@st.cache_resource
def _carrega_retriever_service() -> RetrieverService:
    """Carrega e cacheia MedicalRetriever com ChromaDB."""
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

    try:
        logger.info("Carregando ChromaDB...")
        vector_store = MedicalVectorStore(persist_directory=chroma_dir)
        retriever = MedicalRetriever(vector_store=vector_store)
        logger.info("ChromaDB carregado com sucesso")
        return retriever
    except Exception as e:
        logger.error("Erro ao carregar retriever: %s", e)
        raise


def get_llm_service(required: bool = False) -> LLMService | None:
    """Obtém LLM service com health check."""
    try:
        return _carrega_llm_service()
    except Exception as e:
        if required:
            st.error(f"❌ Erro ao carregar modelo Ollama: {e}")
            st.info(
                "⚠️ Certifique-se de que:\n"
                f"  1. Ollama está rodando em {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}\n"
                f"  2. Modelo {os.getenv('OLLAMA_MODEL', 'llama3')} está disponível"
            )
        return None


def get_retriever_service(required: bool = False) -> RetrieverService | None:
    """Obtém retriever service com health check."""
    try:
        return _carrega_retriever_service()
    except Exception as e:
        if required:
            st.error(f"❌ Erro ao carregar ChromaDB: {e}")
            st.info(
                "⚠️ Certifique-se de que:\n"
                f"  1. ChromaDB está em {os.getenv('CHROMA_PERSIST_DIR', 'chroma_db')}\n"
                f"  2. Você rodou 'python app.py --index-knowledge' antes"
            )
        return None


# ==============================================================================
# Inicialização de Session State
# ==============================================================================


def _inicializa_session_state() -> None:
    """Inicializa variáveis de session state do Streamlit."""
    if "historico_execucoes" not in st.session_state:
        st.session_state.historico_execucoes = []

    if "chat_historico" not in st.session_state:
        st.session_state.chat_historico = []

    if "pacientes_carregados" not in st.session_state:
        st.session_state.pacientes_carregados = {}


_inicializa_session_state()


# ==============================================================================
# Funções de Renderização UI
# ==============================================================================


def _renderiza_header() -> None:
    """Renderiza cabeçalho principal."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            "<div class='main-header'>"
            "<h1>🏥 MedAssist</h1>"
            "<p><em>Assistente Médico Virtual com IA & LangChain</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()


def _renderiza_sidebar() -> str:
    """
    Renderiza sidebar com seleção de pipeline.

    Returns:
        Pipeline selecionado (str)
    """
    with st.sidebar:
        st.markdown("## 📋 Menu")

        pipeline = st.radio(
            "Selecione um pipeline:",
            options=[
                "📝 Perguntas Clínicas",
                "🔄 Fluxo Clínico",
                "📊 Dados & Pré-processamento",
                "📈 Avaliação",
                "⚙️ Configurações",
            ],
            key="pipeline_selecionado",
        )

        st.divider()

        # Expandable: Status de Serviços
        with st.expander("🔌 Status de Serviços"):
            col1, col2 = st.columns(2)

            with col1:
                try:
                    llm = _carrega_llm_service()
                    if llm and llm.is_loaded:
                        st.success("✅ Ollama conectado")
                    else:
                        st.warning("⚠️ Ollama não carregado")
                except Exception:
                    st.error("❌ Ollama indisponível")

            with col2:
                try:
                    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
                    if Path(chroma_dir).exists():
                        st.success("✅ ChromaDB pronto")
                    else:
                        st.warning("⚠️ ChromaDB vazio")
                except Exception:
                    st.error("❌ ChromaDB erro")

    return pipeline


def _renderiza_form_pergunta() -> dict[str, Any] | None:
    """
    Renderiza formulário para pergunta clínica.

    Returns:
        Dict com inputs ou None
    """
    st.subheader("📝 Fazer Pergunta Clínica")

    col1, col2 = st.columns([3, 1])

    with col1:
        pergunta = st.text_area(
            "Pergunta médica:",
            placeholder="Ex: Quais são os efeitos colaterais da metformina?",
            height=100,
            key="pergunta_clinica",
        )

    with col2:
        prioridade = st.selectbox(
            "Prioridade:",
            options=["regular", "urgente", "critica"],
            index=0,
        )

    # Seção expansível: Contexto de Paciente (opcional)
    with st.expander("👤 Contexto do Paciente (opcional)"):
        paciente_id = st.text_input(
            "ID do Paciente:",
            placeholder="Ex: PAC-001",
            key="paciente_id_pergunta",
        )

        contexto = st.text_area(
            "Contexto adicional:",
            placeholder="Idade, diagnósticos, medicamentos, etc.",
            key="contexto_paciente",
        )

    if st.button("🚀 Enviar Pergunta", key="btn_pergunta", use_container_width=True):
        if not pergunta.strip():
            st.warning("⚠️ Digite uma pergunta!")
            return None

        return {
            "pergunta": pergunta,
            "prioridade": prioridade,
            "paciente_id": paciente_id if paciente_id.strip() else None,
            "contexto": contexto,
        }

    return None


def _renderiza_form_fluxo_clinico() -> dict[str, Any] | None:
    """
    Renderiza formulário para fluxo clínico.

    Returns:
        Dict com inputs ou None
    """
    st.subheader("🔄 Fluxo Clínico Completo")

    # Carregador de pacientes
    paciente_dir = Path(os.getenv("PACIENTES_DIR", "data/patients"))
    paciente_files = sorted(paciente_dir.glob("*.json")) if paciente_dir.exists() else []

    if not paciente_files:
        st.warning("⚠️ Nenhum paciente disponível. Use 'Dados & Pré-processamento' → 'Gerar Pacientes' primeiro.")
        return None

    col1, col2 = st.columns(2)

    with col1:
        paciente_selecionado = st.selectbox(
            "Selecione paciente:",
            options=[f.name for f in paciente_files],
            key="paciente_selecionado_fluxo",
        )

    with col2:
        pergunta_fluxo = st.text_input(
            "Pergunta clínica para o fluxo:",
            value="Avaliação clínica geral",
            key="pergunta_fluxo",
        )

    if st.button("🚀 Executar Fluxo", key="btn_fluxo", use_container_width=True):
        if not paciente_selecionado:
            st.warning("⚠️ Selecione um paciente!")
            return None

        try:
            paciente_path = paciente_dir / paciente_selecionado
            with open(paciente_path, "r", encoding="utf-8") as f:
                paciente_data = json.load(f)

            return {
                "paciente_data": paciente_data,
                "pergunta": pergunta_fluxo,
                "paciente_arquivo": paciente_selecionado,
            }
        except Exception as e:
            st.error(f"❌ Erro ao carregar paciente: {e}")
            return None

    return None


def _renderiza_form_dados() -> tuple[str, dict[str, Any]] | None:
    """
    Renderiza formulários para operações de dados.

    Returns:
        Tuple (operação, params) ou None
    """
    st.subheader("📊 Dados & Pré-processamento")

    operacao = st.selectbox(
        "Operação:",
        options=[
            "Pré-processar PubMedQA",
            "Indexar Conhecimento (ChromaDB)",
            "Gerar Pacientes Sintéticos",
        ],
        key="operacao_dados",
    )

    st.divider()

    if operacao == "Pré-processar PubMedQA":
        st.write("**Converte PubMedQA em splits para treinamento**")

        col1, col2, col3 = st.columns(3)

        with col1:
            pubmedqa_file = st.text_input(
                "Arquivo PubMedQA:",
                value="DataSets/ori_pqal.json",
                key="pubmedqa_file",
            )

        with col2:
            ground_truth_file = st.text_input(
                "Ground Truth:",
                value="DataSets/test_ground_truth.json",
                key="ground_truth_file",
            )

        with col3:
            output_dir = st.text_input(
                "Diretório saída:",
                value="data/processed",
                key="preprocess_output",
            )

        if st.button("▶️ Pré-processar", key="btn_preprocess", use_container_width=True):
            return ("preprocessamento", {
                "pubmedqa_file": pubmedqa_file,
                "ground_truth_file": ground_truth_file,
                "output_dir": output_dir,
            })

    elif operacao == "Indexar Conhecimento (ChromaDB)":
        st.write("**Indexa documentos RAG no ChromaDB para buscas semânticas**")

        col1, col2 = st.columns(2)

        with col1:
            rag_file = st.text_input(
                "Arquivo RAG JSONL:",
                value="data/processed/medquad_rag.jsonl",
                key="rag_file",
            )

        with col2:
            chroma_dir = st.text_input(
                "Diretório ChromaDB:",
                value="chroma_db",
                key="chroma_dir_input",
            )

        if st.button("▶️ Indexar", key="btn_indexar", use_container_width=True):
            return ("indexacao", {
                "rag_file": rag_file,
                "chroma_dir": chroma_dir,
            })

    elif operacao == "Gerar Pacientes Sintéticos":
        st.write("**Cria pacientes fictícios para testes do fluxo clínico**")

        col1, col2 = st.columns(2)

        with col1:
            count = st.number_input(
                "Quantidade de pacientes:",
                min_value=1,
                max_value=100,
                value=10,
                key="pacientes_count",
            )

        with col2:
            output_dir_pac = st.text_input(
                "Diretório saída:",
                value="data/patients",
                key="pacientes_output",
            )

        if st.button("▶️ Gerar", key="btn_gerar_pac", use_container_width=True):
            return ("geracao_pacientes", {
                "count": count,
                "output_dir": output_dir_pac,
            })

    return None


def _renderiza_form_avaliacao() -> tuple[str, dict[str, Any]] | None:
    """
    Renderiza formulários para avaliação.

    Returns:
        Tuple (modo, params) ou None
    """
    st.subheader("📈 Avaliação & Métricas")

    modo = st.selectbox(
        "Modo de avaliação:",
        options=[
            "Benchmark Quantitativo",
            "Teste com LLM-as-Judge",
            "Relatório Consolidado",
        ],
        key="modo_avaliacao",
    )

    st.divider()

    if modo == "Benchmark Quantitativo":
        st.write("**Avalia modelo com métricas quantitativas (Accuracy, F1, etc.)**")

        col1, col2, col3 = st.columns(3)

        with col1:
            test_file = st.text_input(
                "Arquivo teste JSONL:",
                value="data/processed/dataset_test.jsonl",
                key="benchmark_test_file",
            )

        with col2:
            max_samples = st.number_input(
                "Máximo de amostras (0=todas):",
                min_value=0,
                value=50,
                key="benchmark_max_samples",
            )

        with col3:
            eval_output = st.text_input(
                "Dir. de saída:",
                value="evaluation_results",
                key="benchmark_output",
            )

        if st.button("▶️ Executar Benchmark", key="btn_benchmark", use_container_width=True):
            return ("benchmark", {
                "test_file": test_file,
                "max_samples": max_samples if max_samples > 0 else None,
                "output_dir": eval_output,
            })

    elif modo == "Teste com LLM-as-Judge":
        st.write("**Avaliação qualitativa com GPT-4o-mini como juiz**")

        col1, col2, col3 = st.columns(3)

        with col1:
            test_file_judge = st.text_input(
                "Arquivo teste JSONL:",
                value="data/processed/dataset_test.jsonl",
                key="judge_test_file",
            )

        with col2:
            judge_model = st.selectbox(
                "Modelo juiz:",
                options=["gpt-4o-mini", "gpt-4o"],
                key="judge_model_select",
            )

        with col3:
            max_judge_samples = st.number_input(
                "Amostras:",
                min_value=1,
                value=30,
                key="judge_max_samples",
            )

        if st.button("▶️ Executar Judge", key="btn_judge", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OPENAI_API_KEY não configurada. Defina no .env")
                return None

            return ("llm_judge", {
                "test_file": test_file_judge,
                "judge_model": judge_model,
                "max_samples": max_judge_samples,
                "output_dir": "evaluation_results",
            })

    elif modo == "Relatório Consolidado":
        st.write("**Consolida resultados de benchmarks anteriores**")

        eval_output_report = st.text_input(
            "Diretório com resultados:",
            value="evaluation_results",
            key="report_eval_output",
        )

        if st.button("▶️ Gerar Relatório", key="btn_report", use_container_width=True):
            return ("relatorio", {
                "eval_output_dir": eval_output_report,
            })

    return None


def _renderiza_configuracoes() -> None:
    """Renderiza página de configurações."""
    st.subheader("⚙️ Configurações")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Variáveis de Ambiente")
        st.code(
            f"""OLLAMA_MODEL={os.getenv('OLLAMA_MODEL', 'llama3')}
OLLAMA_BASE_URL={os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}
CHROMA_PERSIST_DIR={os.getenv('CHROMA_PERSIST_DIR', 'chroma_db')}
OPENAI_API_KEY={'[CONFIGURADA]' if os.getenv('OPENAI_API_KEY') else '[NÃO CONFIGURADA]'}""",
            language="bash",
        )

    with col2:
        st.markdown("### Verificações")
        st.write("**Olhando...**")

        # Verifica Ollama
        try:
            llm = _carrega_llm_service()
            st.success("✅ Ollama está acessível e modelo carregado")
        except Exception as e:
            st.error(f"❌ Ollama: {e}")

        # Verifica ChromaDB
        chroma_path = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
        if chroma_path.exists():
            st.success("✅ Diretório ChromaDB existe")
        else:
            st.warning("⚠️ ChromaDB não inicializado (rode --index-knowledge)")

        # Verifica arquivos de dados
        if Path("data/processed/dataset_test.jsonl").exists():
            st.success("✅ Dataset de teste disponível")
        else:
            st.warning("⚠️ Dataset não encontrado (rode --preprocess)")

        if Path("data/patients").exists() and list(Path("data/patients").glob("*.json")):
            st.success("✅ Pacientes sintéticos disponíveis")
        else:
            st.warning("⚠️ Pacientes não gerados (rode 'Gerar Pacientes Sintéticos')")

    st.divider()

    st.markdown("### Documentação")
    st.markdown("""
    - **Pipeline Pergunta Clínica**: Faz uma pergunta ao modelo com contexto RAG
    - **Fluxo Clínico**: Executa orquestração completa para um paciente
    - **Pré-processamento**: Converte PubMedQA em splits de treino/val/teste
    - **Indexação**: Indexa documentos RAG no ChromaDB
    - **Geração de Pacientes**: Cria dados sintéticos para testes
    - **Benchmark**: Avalia modelo com métricas quantitativas
    - **LLM-as-Judge**: Avaliação qualitativa com GPT-4o-mini
    """)


def _renderiza_resultados(resultado: dict[str, Any], tipo_pipeline: str) -> None:
    """
    Renderiza resultados da execução.

    Args:
        resultado: Dict com resultado
        tipo_pipeline: Tipo de pipeline executado
    """
    if not resultado.get("sucesso"):
        st.error(f"❌ Erro: {resultado.get('erro', 'Erro desconhecido')}")

        if "tipo_erro" in resultado:
            with st.expander("📋 Detalhes técnicos"):
                st.code(resultado.get("tipo_erro", ""))

        return

    st.success("✅ Execução bem-sucedida!")

    # Renderiza por tipo
    if tipo_pipeline == "pergunta_clinica":
        st.markdown("### Resposta")
        st.write(resultado.get("resposta", ""))

        col1, col2, col3 = st.columns(3)
        with col1:
            confianca = resultado.get("confianca", 0.5)
            conf_color = "🟢" if confianca >= 0.7 else ("🟡" if confianca >= 0.4 else "🔴")
            st.metric("Confiança", f"{confianca:.0%}", f"{conf_color}")

        with col2:
            st.metric("Modelo", resultado.get("modelo", "?"))

        with col3:
            st.metric("Timestamp", resultado.get("timestamp", "")[:10])

        if resultado.get("fontes"):
            st.markdown("### 📚 Fontes")
            for i, fonte in enumerate(resultado.get("fontes", [])[:5], 1):
                st.write(f"{i}. {fonte}")

        if resultado.get("disclaimer"):
            st.info(f"ℹ️ **Disclaimer**: {resultado.get('disclaimer')}")

        if resultado.get("guardrails"):
            st.warning(f"⚠️ Guardrails acionados: {', '.join(resultado.get('guardrails', []))}")

    elif tipo_pipeline == "fluxo_clinico":
        triage_level = resultado.get("triagem_nivel", "?")
        triage_colors = {
            "critico": "🔴",
            "urgente": "🟠",
            "regular": "🟢",
        }
        icon = triage_colors.get(triage_level, "⚪")

        st.markdown(f"### {icon} Triagem: {triage_level.upper()}")
        st.write(resultado.get("triagem_descricao", ""))

        col1, col2 = st.columns(2)

        with col1:
            exames_pend = resultado.get("exames_pendentes", [])
            if exames_pend:
                st.markdown("#### Exames Pendentes")
                for exam in exames_pend:
                    st.write(f"• {exam}")

            exames_sug = resultado.get("exames_sugeridos", [])
            if exames_sug:
                st.markdown("#### Exames Sugeridos")
                for exam in exames_sug:
                    st.write(f"• {exam}")

        with col2:
            if resultado.get("tratamento"):
                st.markdown("#### Sugestão de Tratamento")
                st.write(resultado.get("tratamento"))

        alertas = resultado.get("alertas", [])
        if alertas:
            st.markdown("#### ⚠️ Alertas")
            for alerta in alertas:
                severity = alerta.get("severity", "info")
                msg = alerta.get("message", "")
                if severity == "critica":
                    st.error(f"🔴 {msg}")
                elif severity == "alta":
                    st.warning(f"🟠 {msg}")
                elif severity == "media":
                    st.info(f"🟡 {msg}")
                else:
                    st.info(f"🟢 {msg}")

        if resultado.get("validacao_humana_necessaria"):
            st.warning(_generate_alert(
                "⚠️ VALIDAÇÃO HUMANA NECESSÁRIA",
                resultado.get("motivo_validacao", ""),
            ))

    elif tipo_pipeline in ["preprocessamento", "indexacao", "geracao_pacientes"]:
        col1, col2, col3 = st.columns(3)

        if "pubmedqa_entries" in resultado:
            with col1:
                st.metric("Entradas PubMedQA", resultado.get("pubmedqa_entries"))
            with col2:
                st.metric("Documentos RAG", resultado.get("rag_documents"))
            with col3:
                st.metric("Output Dir", resultado.get("output_dir", "")[:30] + "...")

            train_samples = resultado.get("train_samples", 0)
            val_samples = resultado.get("val_samples", 0)
            test_samples = resultado.get("test_samples", 0)

            if train_samples > 0:
                st.markdown("### Dataset Splits")
                data = {
                    "Train": train_samples,
                    "Validation": val_samples,
                    "Test": test_samples,
                }
                st.bar_chart(data)

        elif "documentos_indexados" in resultado:
            with col1:
                st.metric("Documentos Indexados", resultado.get("documentos_indexados"))
            with col2:
                st.metric("ChromaDB Dir", resultado.get("chroma_dir", ""))

        elif "pacientes_gerados" in resultado:
            with col1:
                st.metric("Pacientes Gerados", resultado.get("pacientes_gerados"))
            with col2:
                st.metric("Output Dir", resultado.get("output_dir", ""))

    elif tipo_pipeline == "benchmark":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{resultado.get('accuracy', 0):.4f}")
        with col2:
            st.metric("F1 Macro", f"{resultado.get('f1_macro', 0):.4f}")
        with col3:
            st.metric("Exact Match", f"{resultado.get('exact_match', 0):.4f}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Token F1", f"{resultado.get('token_f1', 0):.4f}")
        with col2:
            st.metric("Tempo Inference (s)", f"{resultado.get('tempo_inference_s', 0):.1f}")
        with col3:
            st.metric("Amostras", resultado.get("amostras_avaliadas"))

    elif tipo_pipeline == "llm_judge":
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Amostras Avaliadas", resultado.get("amostras_avaliadas"))
            st.metric("Accuracy", f"{resultado.get('accuracy', 0):.4f}")

        with col2:
            st.metric("Exact Match", f"{resultado.get('exact_match', 0):.4f}")
            st.metric("Token F1", f"{resultado.get('token_f1', 0):.4f}")

        if "judge_summary" in resultado:
            st.markdown("### Judge Summary")
            st.write(resultado.get("judge_summary"))

    elif tipo_pipeline == "relatorio":
        resultados = resultado.get("resultados", [])

        if not resultados:
            st.info("ℹ️ " + resultado.get("mensagem", "Nenhum resultado encontrado"))
        else:
            st.markdown("### Benchmarks Realizados")

            # Tabela
            import pandas as pd

            df = pd.DataFrame(resultados)
            st.dataframe(df, use_container_width=True)

    st.divider()

    # Adiciona ao histórico
    _adiciona_ao_historico(tipo_pipeline, resultado)


def _adiciona_ao_historico(tipo_pipeline: str, resultado: dict[str, Any]) -> None:
    """Adiciona execução ao histórico de sessão."""
    entrada_resumida = ""

    if tipo_pipeline == "pergunta_clinica":
        entrada_resumida = f"Pergunta: {resultado.get('resposta', '')[:50]}..."
    elif tipo_pipeline == "fluxo_clinico":
        entrada_resumida = f"Triagem: {resultado.get('triagem_nivel', '?')}"
    else:
        entrada_resumida = f"{tipo_pipeline}"

    st.session_state.historico_execucoes.append({
        "timestamp": datetime.now().isoformat(),
        "pipeline": tipo_pipeline,
        "entrada": entrada_resumida,
        "status": "✅ Sucesso" if resultado.get("sucesso") else "❌ Erro",
        "resultado": str(resultado.get("sucesso", False)),
    })


def _renderiza_historico() -> None:
    """Renderiza tabela de histórico de execuções."""
    st.subheader("📜 Histórico de Execuções (Sessão)")

    if not st.session_state.historico_execucoes:
        st.info("ℹ️ Nenhuma execução ainda.")
        return

    import pandas as pd

    historico_df = pd.DataFrame(st.session_state.historico_execucoes)
    historico_df = historico_df[["timestamp", "pipeline", "status"]]

    st.dataframe(historico_df, use_container_width=True, hide_index=True)

    # Download CSV
    csv = historico_df.to_csv(index=False)
    st.download_button(
        "📥 Baixar Histórico (CSV)",
        data=csv,
        file_name=f"historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def _generate_alert(title: str, message: str) -> str:
    """Gera HTML para alerta customizado."""
    return f"**{title}**\n\n{message}"


# ==============================================================================
# Execução de Pipelines
# ==============================================================================


def _executa_pergunta_clinica(params: dict[str, Any]) -> None:
    """Executa pipeline de pergunta clínica."""
    llm_service = get_llm_service(required=True)
    retriever_service = get_retriever_service(required=False)

    if not llm_service:
        return

    with st.spinner("🔄 Processando pergunta..."):
        adaptador = AdaptadorPerguntaClinica(
            llm_service=llm_service,
            retriever_service=retriever_service,
        )

        resultado = adaptador.executar(
            pergunta=params["pergunta"],
            paciente_id=params.get("paciente_id"),
            contexto_paciente=params.get("contexto", ""),
            prioridade=params.get("prioridade", "regular"),
        )

    _renderiza_resultados(resultado, "pergunta_clinica")


def _executa_fluxo_clinico(params: dict[str, Any]) -> None:
    """Executa pipeline de fluxo clínico."""
    llm_service = get_llm_service(required=True)
    retriever_service = get_retriever_service(required=False)

    if not llm_service:
        return

    with st.spinner("🔄 Executando fluxo clínico..."):
        adaptador = AdaptadorFluxoClinico(
            llm_service=llm_service,
            retriever_service=retriever_service,
        )

        resultado = adaptador.executar(params["paciente_data"])

    _renderiza_resultados(resultado, "fluxo_clinico")


def _executa_preprocessamento(params: dict[str, Any]) -> None:
    """Executa pré-processamento."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("📋 Processando PubMedQA...")
    progress_bar.progress(33)

    resultado = AdaptadorPreprocessamento.executar(
        pubmedqa_file=params["pubmedqa_file"],
        ground_truth_file=params["ground_truth_file"],
        output_dir=params["output_dir"],
    )

    progress_bar.progress(100)
    status_text.text("✅ Concluído!")

    _renderiza_resultados(resultado, "preprocessamento")


def _executa_indexacao(params: dict[str, Any]) -> None:
    """Executa indexação de conhecimento."""
    with st.spinner("🔄 Indexando documentos..."):
        resultado = AdaptadorIndexacao.executar(
            rag_file=params["rag_file"],
            chroma_dir=params["chroma_dir"],
        )

    _renderiza_resultados(resultado, "indexacao")


def _executa_geracao_pacientes(params: dict[str, Any]) -> None:
    """Executa geração de pacientes sintéticos."""
    with st.spinner(f"🔄 Gerando {params['count']} pacientes..."):
        resultado = AdaptadorGeracaoPacientes.executar(
            count=params["count"],
            output_dir=params["output_dir"],
        )

    _renderiza_resultados(resultado, "geracao_pacientes")
    st.session_state.clear()  # Limpa cache de pacientes para recarregar


def _executa_benchmark(params: dict[str, Any]) -> None:
    """Executa benchmark quantitativo."""
    llm_service = get_llm_service(required=True)

    if not llm_service:
        return

    with st.spinner("⏳ Executando benchmark (pode levar vários minutos)..."):
        adaptador = AdaptadorBenchmark(llm_service=llm_service)

        resultado = adaptador.executar(
            test_file=params["test_file"],
            max_samples=params.get("max_samples"),
            output_dir=params.get("output_dir", "evaluation_results"),
        )

    _renderiza_resultados(resultado, "benchmark")


def _executa_llm_judge(params: dict[str, Any]) -> None:
    """Executa avaliação com LLM-as-judge."""
    llm_service = get_llm_service(required=True)

    if not llm_service:
        return

    with st.spinner("⏳ Executando avaliação com judge (pode levar vários minutos)..."):
        adaptador = AdaptadorAvaliadorLLM(llm_service=llm_service)

        resultado = adaptador.executar(
            test_file=params["test_file"],
            judge_model=params.get("judge_model", "gpt-4o-mini"),
            max_samples=params.get("max_samples", 50),
            output_dir=params.get("output_dir", "evaluation_results"),
        )

    _renderiza_resultados(resultado, "llm_judge")


def _executa_relatorio(params: dict[str, Any]) -> None:
    """Executa geração de relatório consolidado."""
    resultado = AdaptadorRelatorioAvaliacao.executar(
        eval_output_dir=params["eval_output_dir"],
    )

    _renderiza_resultados(resultado, "relatorio")


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    """Função principal da aplicação Streamlit."""
    _renderiza_header()
    pipeline = _renderiza_sidebar()

    if pipeline == "📝 Perguntas Clínicas":
        params = _renderiza_form_pergunta()
        if params:
            _executa_pergunta_clinica(params)

    elif pipeline == "🔄 Fluxo Clínico":
        params = _renderiza_form_fluxo_clinico()
        if params:
            _executa_fluxo_clinico(params)

    elif pipeline == "📊 Dados & Pré-processamento":
        resultado = _renderiza_form_dados()
        if resultado:
            tipo, params = resultado

            if tipo == "preprocessamento":
                _executa_preprocessamento(params)
            elif tipo == "indexacao":
                _executa_indexacao(params)
            elif tipo == "geracao_pacientes":
                _executa_geracao_pacientes(params)

    elif pipeline == "📈 Avaliação":
        resultado = _renderiza_form_avaliacao()
        if resultado:
            tipo, params = resultado

            if tipo == "benchmark":
                _executa_benchmark(params)
            elif tipo == "llm_judge":
                _executa_llm_judge(params)
            elif tipo == "relatorio":
                _executa_relatorio(params)

    elif pipeline == "⚙️ Configurações":
        _renderiza_configuracoes()

    st.divider()
    _renderiza_historico()


if __name__ == "__main__":
    main()
