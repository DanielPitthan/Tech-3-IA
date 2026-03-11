"""
QUICK START — Guia de 5 minutos para colocar a UI em funcionamento.

1. Prepare o ambiente (1 min)
2. Inicie os serviços (2 min)
3. Rode a interface (2 min)
"""

import subprocess
import sys
from pathlib import Path


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def check_requirements():
    """Verifica se dependências estão instaladas."""
    print_section("📋 PASSO 1: Verificar Dependências")

    required = ["streamlit", "torch", "transformers", "langchain"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  Pacotes faltando: {', '.join(missing)}")
        print(f"\n   Instale com:")
        print(f"   pip install -r requirements.txt")
        return False

    print("\n✅ Todas as dependências estão instaladas")
    return True


def check_ollama():
    """Verifica se Ollama está rodando."""
    print_section("🔌 PASSO 2: Verificar Ollama")

    import os

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    try:
        import requests

        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code == 200:
            print(f"  ✅ Ollama rodando em {ollama_url}")

            models = response.json().get("models", [])
            installed_models = [m["name"].split(":")[0] for m in models]

            if ollama_model in installed_models or any(ollama_model in m for m in installed_models):
                print(f"  ✅ Modelo '{ollama_model}' disponível")
                return True
            else:
                print(f"  ⚠️  Modelo '{ollama_model}' não encontrado")
                print(f"\n     Puxe o modelo com:")
                print(f"     ollama pull {ollama_model}")
                return False

    except Exception as e:
        print(f"  ❌ Ollama não está rodando")
        print(f"\n     Inicie em outro terminal:")
        print(f"     ollama serve")
        print(f"\n     Detalhes do erro: {e}")
        return False


def check_chroma():
    """Verifica se ChromaDB está inicializado."""
    print_section("💾 PASSO 3: Verificar ChromaDB")

    import os

    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    chroma_path = Path(chroma_dir)

    if chroma_path.exists():
        print(f"  ✅ Diretório ChromaDB existe: {chroma_dir}")
        return True
    else:
        print(f"  ⚠️  ChromaDB não inicializado")
        print(f"\n     Rode em outro terminal:")
        print(f"     python app.py --index-knowledge")
        return False


def check_datasets():
    """Verifica se datasets existem."""
    print_section("📊 PASSO 4: Verificar Datasets")

    datasets = {
        "Teste": "data/processed/dataset_test.jsonl",
        "Treino": "data/processed/dataset_train.jsonl",
        "Validação": "data/processed/dataset_val.jsonl",
    }

    all_exist = True

    for name, path in datasets.items():
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name} não encontrado: {path}")
            all_exist = False

    if not all_exist:
        print(f"\n     Gere datasets com:")
        print(f"     python app.py --preprocess")

    return all_exist


def check_patients():
    """Verifica se pacientes sintéticos existem."""
    print_section("👥 PASSO 5: Verificar Pacientes")

    paciente_dir = Path("data/patients")

    if paciente_dir.exists():
        pacientes = list(paciente_dir.glob("*.json"))
        if pacientes:
            print(f"  ✅ Pacientes encontrados: {len(pacientes)}")
            return True
        else:
            print(f"  ⚠️  Nenhum paciente em {paciente_dir}")
    else:
        print(f"  ⚠️  Diretório não existe: {paciente_dir}")

    print(f"\n     Gere pacientes com:")
    print(f"     python app.py --generate-patients --count 20")
    return False


def summarize_status(results):
    """Resumo do status."""
    print_section("📊 RESUMO")

    print("Status dos serviços:")
    print(f"  {'Dependências':<20} {'✅' if results['deps'] else '❌'}")
    print(f"  {'Ollama':<20} {'✅' if results['ollama'] else '❌'}")
    print(f"  {'ChromaDB':<20} {'✅' if results['chroma'] else '⚠️ ' if isinstance(results['chroma'], bool) else '❌'}")
    print(f"  {'Datasets':<20} {'✅' if results['datasets'] else '⚠️'}")
    print(f"  {'Pacientes':<20} {'✅' if results['patients'] else '⚠️'}")

    essential_ok = results["deps"] and results["ollama"]

    print(f"\n{'=' * 60}\n")

    if essential_ok:
        print("✅ Serviços essenciais OK! Pronto para usar a interface.\n")
        print("   PRÓXIMO PASSO:")
        print("   $ streamlit run streamlit_app.py\n")
    else:
        print("❌ Serviços essenciais faltando. Veja instruções acima.\n")

    return essential_ok


def main():
    """Executa verificações."""
    print("\n" + "🚀 QUICK START — MedAssist UI" + "\n")

    results = {
        "deps": check_requirements(),
        "ollama": check_ollama(),
        "chroma": check_chroma(),
        "datasets": check_datasets(),
        "patients": check_patients(),
    }

    essential_ok = summarize_status(results)

    if essential_ok:
        print("=" * 60)
        print("\nDeseja iniciar a interface agora? (s/n)")
        response = input(">>> ").strip().lower()

        if response == "s":
            print("\n⏳ Iniciando Streamlit...")
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
                cwd=Path(__file__).parent,
            )
        else:
            print("\n✅ Para iniciar depois, execute:")
            print("   streamlit run streamlit_app.py\n")


if __name__ == "__main__":
    main()
