"""Permite executar o pacote via: python -m medical_assistant [opções]"""

import sys
from pathlib import Path

# Garantir que a raiz do projeto esteja no sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import main

if __name__ == "__main__":
    main()
