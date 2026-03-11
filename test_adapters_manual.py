"""
Exemplos de teste dos adaptadores (sem Streamlit).

Útil para debugging e testes unitários.

Execute com:
    python test_adapters_manual.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Importar adaptadores
from medical_assistant.interfaces.ui_adapter import (
    AdaptadorPerguntaClinica,
    AdaptadorFluxoClinico,
    AdaptadorPreprocessamento,
    AdaptadorIndexacao,
    AdaptadorGeracaoPacientes,
    AdaptadorBenchmark,
    AdaptadorRelatorioAvaliacao,
)

# Importar serviços
from medical_assistant.infrastructure.llm.ollama_model import OllamaModelAdapter
from medical_assistant.infrastructure.persistence.vector_store import MedicalVectorStore
from medical_assistant.infrastructure.langchain.retrievers import MedicalRetriever

load_dotenv()


def test_servicos():
    """Testa carregamento de serviços."""
    print("\n" + "=" * 60)
    print("TEST 1: Carregamento de Serviços")
    print("=" * 60)

    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    try:
        print(f"🔄 Carregando Ollama ({ollama_model}) em {ollama_url}...")
        llm_service = OllamaModelAdapter(
            model_name=ollama_model,
            base_url=ollama_url,
        )
        llm_service.load()
        print(f"✅ Ollama carregado | is_loaded={llm_service.is_loaded}")
    except Exception as e:
        print(f"❌ Erro ao carregar Ollama: {e}")
        return False

    try:
        print(f"🔄 Carregando ChromaDB...")
        chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        vector_store = MedicalVectorStore(persist_directory=chroma_dir)
        retriever = MedicalRetriever(vector_store=vector_store)
        print(f"✅ ChromaDB carregado | dir={chroma_dir}")
    except Exception as e:
        print(f"❌ Erro ao carregar ChromaDB: {e}")
        return False

    return True, llm_service, retriever


def test_pergunta_clinica(llm_service, retriever):
    """Testa adaptador de pergunta clínica."""
    print("\n" + "=" * 60)
    print("TEST 2: Pergunta Clínica")
    print("=" * 60)

    adaptador = AdaptadorPerguntaClinica(
        llm_service=llm_service,
        retriever_service=retriever,
    )

    pergunta = "Quais são os efeitos colaterais da aspirina?"

    print(f"❓ Pergunta: {pergunta}")
    print("🔄 Processando...")

    resultado = adaptador.executar(
        pergunta=pergunta,
        patient_id=None,
        contexto_paciente="",
        prioridade="regular",
    )

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Confiança: {resultado.get('confianca', 0):.0%}")
        print(f"   Fontes: {len(resultado.get('fontes', []))} documentos")
        print(f"   Resposta (primeiros 200 chars):")
        print(f"   {resultado.get('resposta', '')[:200]}...")
        return True
    else:
        print(f"❌ Erro: {resultado.get('erro')}")
        return False


def test_fluxo_clinico(llm_service, retriever):
    """Testa adaptador de fluxo clínico."""
    print("\n" + "=" * 60)
    print("TEST 3: Fluxo Clínico")
    print("=" * 60)

    # Procurar um paciente disponível
    paciente_dir = Path("data/patients")
    if not paciente_dir.exists():
        print(f"⚠️  Nenhum paciente encontrado em {paciente_dir}")
        print("   Gere pacientes com: python app.py --generate-patients")
        return None

    paciente_files = list(paciente_dir.glob("*.json"))
    if not paciente_files:
        print(f"⚠️  Nenhum arquivo de paciente encontrado")
        return None

    paciente_file = paciente_files[0]
    print(f"📋 Novo paciente: {paciente_file.name}")

    with open(paciente_file, "r", encoding="utf-8") as f:
        paciente_data = json.load(f)

    adaptador = AdaptadorFluxoClinico(
        llm_service=llm_service,
        retriever_service=retriever,
    )

    print("🔄 Executando fluxo clínico...")

    resultado = adaptador.executar(paciente_data)

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Triagem: {resultado.get('triagem_nivel', '?')}")
        print(f"   Alertas: {len(resultado.get('alertas', []))}")
        print(f"   Validação humana necessária: {resultado.get('validacao_humana_necessaria')}")
        return True
    else:
        print(f"❌ Erro: {resultado.get('erro')}")
        return False


def test_preprocessamento():
    """Testa adaptador de pré-processamento."""
    print("\n" + "=" * 60)
    print("TEST 4: Pré-processamento")
    print("=" * 60)

    pubmedqa_file = "DataSets/ori_pqal.json"
    ground_truth_file = "DataSets/test_ground_truth.json"
    output_dir = "data/processed"

    if not Path(pubmedqa_file).exists():
        print(f"⚠️  Arquivo não encontrado: {pubmedqa_file}")
        return None

    print(f"📂 PubMedQA: {pubmedqa_file}")
    print(f"📂 Ground Truth: {ground_truth_file}")
    print(f"📂 Output: {output_dir}")
    print("🔄 Processando (pode levar tempo)...")

    resultado = AdaptadorPreprocessamento.executar(
        pubmedqa_file=pubmedqa_file,
        ground_truth_file=ground_truth_file,
        output_dir=output_dir,
    )

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Entradas PubMedQA: {resultado.get('pubmedqa_entries')}")
        print(f"   Documentos RAG: {resultado.get('rag_documents')}")
        print(f"   Train split: {resultado.get('train_samples')} amostras")
        print(f"   Val split: {resultado.get('val_samples')} amostras")
        print(f"   Test split: {resultado.get('test_samples')} amostras")
        return True
    else:
        print(f"❌ Erro: {resultado.get('erro')}")
        return False


def test_indexacao():
    """Testa adaptador de indexação."""
    print("\n" + "=" * 60)
    print("TEST 5: Indexação (ChromaDB)")
    print("=" * 60)

    rag_file = "data/processed/medquad_rag.jsonl"
    chroma_dir = "chroma_db"

    if not Path(rag_file).exists():
        print(f"⚠️  Arquivo não encontrado: {rag_file}")
        print("   Execute TEST 4 (pré-processamento) primeiro")
        return None

    print(f"📂 RAG file: {rag_file}")
    print(f"📂 ChromaDB dir: {chroma_dir}")
    print("🔄 Indexando...")

    resultado = AdaptadorIndexacao.executar(
        rag_file=rag_file,
        chroma_dir=chroma_dir,
    )

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Documentos indexados: {resultado.get('documentos_indexados')}")
        return True
    else:
        print(f"❌ Erro: {resultado.get('erro')}")
        return False


def test_geracao_pacientes():
    """Testa adaptador de geração de pacientes."""
    print("\n" + "=" * 60)
    print("TEST 6: Geração de Pacientes Sintéticos")
    print("=" * 60)

    count = 5
    output_dir = "data/patients"

    print(f"👥 Quantidade: {count}")
    print(f"📂 Output: {output_dir}")
    print("🔄 Gerando...")

    resultado = AdaptadorGeracaoPacientes.executar(
        count=count,
        output_dir=output_dir,
    )

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Pacientes gerados: {resultado.get('pacientes_gerados')}")
        return True
    else:
        print(f"❌ Erro: {resultado.get('erro')}")
        return False


def test_relatorio():
    """Testa adaptador de relatório."""
    print("\n" + "=" * 60)
    print("TEST 7: Relatório de Avaliação")
    print("=" * 60)

    eval_output_dir = "evaluation_results"

    print(f"📂 Diretório: {eval_output_dir}")
    print("🔄 Gerando relatório...")

    resultado = AdaptadorRelatorioAvaliacao.executar(
        eval_output_dir=eval_output_dir,
    )

    if resultado.get("sucesso"):
        print(f"✅ Sucesso!")
        print(f"   Benchmarks encontrados: {resultado.get('total_benchmarks')}")
        if resultado.get("resultados"):
            for res in resultado.get("resultados", []):
                print(f"   - {res.get('arquivo')}: Accuracy={res.get('accuracy'):.4f}")
        return True
    else:
        print(f"⚠️  {resultado.get('erro')}")
        return None


def main():
    """Executa todos os testes."""
    print("\n" + "🧪 TESTES DE ADAPTADORES (sem Streamlit)" + "\n")

    # Testa serviços
    result = test_servicos()
    if result is True:
        print("\n❌ Serviços não carregados corretamente")
        return
    elif result is None:
        return

    _, llm_service, retriever = result

    # Testes de execução
    test_pergunta_clinica(llm_service, retriever)
    test_fluxo_clinico(llm_service, retriever)

    # Testes de dados (comentados por padrão — demoram)
    # test_preprocessamento()
    # test_indexacao()
    # test_geracao_pacientes()

    # Teste de relatório
    test_relatorio()

    print("\n" + "=" * 60)
    print("✅ Testes concluídos!")
    print("=" * 60 + "\n")
    print("Para executar a UI completa:")
    print("  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
