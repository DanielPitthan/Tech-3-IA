"""
CLI Principal — Interface de linha de comando do MedAssist.

Uso:
    python app.py --chat
    python app.py --ask "Pergunta clínica"
    python app.py --finetune
    python app.py --evaluate benchmark
    python app.py --all

Este módulo delega ao app.py na raiz do projeto.
Para uso direto: python app.py [opções]
Alternativa:     python -m medical_assistant [opções]
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Ponto de entrada que delega ao app.py."""
    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
