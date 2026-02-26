"""
Repositório de pacientes baseado em JSON/arquivo.

Implementação simples para demonstração do DDD.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.repositories.patient_repository import PatientRepository

logger = logging.getLogger(__name__)


class JsonPatientRepository(PatientRepository):
    """Repositório de pacientes armazenados em arquivo JSON."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self._patients: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Carrega dados do arquivo JSON."""
        if self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._patients = {p["id"]: p for p in data}
                elif isinstance(data, dict):
                    self._patients = data
            logger.info("Carregados %d pacientes de %s", len(self._patients), self.filepath)
        else:
            logger.warning("Arquivo de pacientes não encontrado: %s", self.filepath)

    def get_by_id(self, patient_id: str) -> Patient | None:
        """Busca paciente por ID."""
        data = self._patients.get(patient_id)
        if data:
            return Patient.from_dict(data)
        return None

    def list_all(self) -> list[Patient]:
        """Lista todos os pacientes."""
        return [Patient.from_dict(p) for p in self._patients.values()]

    def save(self, patient: Patient) -> None:
        """Salva ou atualiza um paciente."""
        self._patients[patient.id] = patient.__dict__
        self._persist()

    def _persist(self) -> None:
        """Persiste dados no arquivo JSON."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(list(self._patients.values()), f, ensure_ascii=False, indent=2, default=str)
