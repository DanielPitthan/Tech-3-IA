"""
Entidade: Paciente.

Representa um paciente no contexto do assistente médico,
com dados anonimizados para processamento pela LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class Patient:
    """Entidade Paciente com dados clínicos."""

    id: str
    nome_anonimizado: str
    idade: int
    sexo: str
    setor: str
    diagnosticos: list[str] = field(default_factory=list)
    medicamentos_em_uso: list[dict[str, str]] = field(default_factory=list)
    exames: list[dict[str, Any]] = field(default_factory=list)
    alergias: list[str] = field(default_factory=list)
    sinais_vitais: dict[str, float] = field(default_factory=dict)
    queixa_principal: str = ""
    data_admissao: date | None = None

    @property
    def has_critical_vitals(self) -> bool:
        """Verifica se os sinais vitais estão em faixas críticas."""
        sv = self.sinais_vitais
        if not sv:
            return False

        return (
            sv.get("pa_sistolica", 120) > 180
            or sv.get("pa_sistolica", 120) < 90
            or sv.get("fc", 80) > 130
            or sv.get("fc", 80) < 40
            or sv.get("spo2", 98) < 90
            or sv.get("temperatura", 37.0) > 39.0
        )

    @property
    def pending_exams(self) -> list[dict[str, Any]]:
        """Retorna exames pendentes."""
        return [e for e in self.exames if e.get("status") == "pendente"]

    @property
    def available_results(self) -> list[dict[str, Any]]:
        """Retorna exames com resultados disponíveis."""
        return [e for e in self.exames if e.get("status") == "resultado_disponivel"]

    def to_clinical_summary(self) -> str:
        """Gera resumo clínico textual para contexto da LLM."""
        parts = [
            f"Paciente: {self.nome_anonimizado}",
            f"Idade: {self.idade} anos | Sexo: {self.sexo}",
            f"Setor: {self.setor}",
            f"Queixa principal: {self.queixa_principal}",
        ]

        if self.diagnosticos:
            parts.append(f"Diagnósticos: {', '.join(self.diagnosticos)}")

        if self.medicamentos_em_uso:
            meds = [f"{m['nome']} {m['dose']} ({m['freq']})" for m in self.medicamentos_em_uso]
            parts.append(f"Medicamentos em uso: {', '.join(meds)}")

        if self.alergias and self.alergias != ["Nenhuma alergia conhecida"]:
            parts.append(f"Alergias: {', '.join(self.alergias)}")

        if self.sinais_vitais:
            sv = self.sinais_vitais
            parts.append(
                f"Sinais vitais: PA {sv.get('pa_sistolica', '?')}/"
                f"{sv.get('pa_diastolica', '?')} mmHg, "
                f"FC {sv.get('fc', '?')} bpm, "
                f"FR {sv.get('fr', '?')} irpm, "
                f"T {sv.get('temperatura', '?')}°C, "
                f"SpO2 {sv.get('spo2', '?')}%"
            )

        pending = self.pending_exams
        if pending:
            exam_names = [e["nome"] for e in pending]
            parts.append(f"Exames pendentes: {', '.join(exam_names)}")

        results = self.available_results
        if results:
            for r in results:
                parts.append(f"Resultado {r['nome']}: {r.get('resultado', 'N/A')}")

        return "\n".join(parts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Patient:
        """Cria Patient a partir de dicionário."""
        admissao = data.get("data_admissao")
        if isinstance(admissao, str):
            admissao = date.fromisoformat(admissao)

        return cls(
            id=data["id"],
            nome_anonimizado=data.get("nome_anonimizado", f"[PACIENTE_{data['id']}]"),
            idade=data.get("idade", 0),
            sexo=data.get("sexo", "N/I"),
            setor=data.get("setor", "N/I"),
            diagnosticos=data.get("diagnosticos", []),
            medicamentos_em_uso=data.get("medicamentos_em_uso", []),
            exames=data.get("exames", []),
            alergias=data.get("alergias", []),
            sinais_vitais=data.get("sinais_vitais", {}),
            queixa_principal=data.get("queixa_principal", ""),
            data_admissao=admissao,
        )
