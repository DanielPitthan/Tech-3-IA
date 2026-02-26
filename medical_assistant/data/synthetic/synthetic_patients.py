"""
Gerador de dados sintéticos de pacientes fictícios.

Utiliza a biblioteca Faker para gerar dados realistas
(nomes, idades, exames, diagnósticos) para demonstração
do fluxo clínico completo do LangGraph.

⚠️ Todos os dados são FICTÍCIOS e NÃO representam pacientes reais.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Dados médicos de referência para geração sintética
DIAGNOSTICOS = [
    "Hipertensão Arterial Sistêmica",
    "Diabetes Mellitus Tipo 2",
    "Pneumonia Adquirida na Comunidade",
    "Insuficiência Cardíaca Congestiva",
    "Infecção do Trato Urinário",
    "Doença Pulmonar Obstrutiva Crônica",
    "Asma Brônquica",
    "Fibrilação Atrial",
    "Insuficiência Renal Crônica",
    "Trombose Venosa Profunda",
    "Anemia Ferropriva",
    "Hipotireoidismo",
    "Gastrite Crônica",
    "Artrite Reumatóide",
    "Depressão Maior",
]

MEDICAMENTOS = [
    {"nome": "Losartana", "dose": "50mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Metformina", "dose": "850mg", "via": "oral", "freq": "2x/dia"},
    {"nome": "Amoxicilina", "dose": "500mg", "via": "oral", "freq": "8/8h"},
    {"nome": "Enalapril", "dose": "10mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Omeprazol", "dose": "20mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Sinvastatina", "dose": "20mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "AAS", "dose": "100mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Insulina NPH", "dose": "10UI", "via": "subcutânea", "freq": "2x/dia"},
    {"nome": "Furosemida", "dose": "40mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Prednisona", "dose": "20mg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Levotiroxina", "dose": "75mcg", "via": "oral", "freq": "1x/dia"},
    {"nome": "Warfarina", "dose": "5mg", "via": "oral", "freq": "1x/dia"},
]

EXAMES = [
    {"nome": "Hemograma Completo", "tipo": "laboratorial"},
    {"nome": "Glicemia de Jejum", "tipo": "laboratorial"},
    {"nome": "HbA1c", "tipo": "laboratorial"},
    {"nome": "Creatinina Sérica", "tipo": "laboratorial"},
    {"nome": "Ureia", "tipo": "laboratorial"},
    {"nome": "TGO/TGP", "tipo": "laboratorial"},
    {"nome": "TSH", "tipo": "laboratorial"},
    {"nome": "T4 Livre", "tipo": "laboratorial"},
    {"nome": "Potássio Sérico", "tipo": "laboratorial"},
    {"nome": "Sódio Sérico", "tipo": "laboratorial"},
    {"nome": "PCR", "tipo": "laboratorial"},
    {"nome": "Raio-X de Tórax", "tipo": "imagem"},
    {"nome": "ECG", "tipo": "cardiológico"},
    {"nome": "Ecocardiograma", "tipo": "cardiológico"},
    {"nome": "Tomografia de Tórax", "tipo": "imagem"},
    {"nome": "Urina Tipo I", "tipo": "laboratorial"},
    {"nome": "Urocultura", "tipo": "laboratorial"},
    {"nome": "INR", "tipo": "laboratorial"},
]

ALERGIAS = [
    "Penicilina",
    "Sulfa",
    "Dipirona",
    "AINEs",
    "Contraste Iodado",
    "Látex",
    "Cefalosporina",
    "Nenhuma alergia conhecida",
]

SETORES = [
    "Pronto-Socorro",
    "Enfermaria Clínica",
    "UTI",
    "Ambulatório",
    "Centro Cirúrgico",
    "Unidade Coronariana",
]


def generate_synthetic_patient(patient_id: int, seed: int | None = None) -> dict[str, Any]:
    """
    Gera um paciente sintético com dados clínicos fictícios.

    Args:
        patient_id: ID sequencial do paciente
        seed: Semente para reprodutibilidade
    """
    if seed is not None:
        random.seed(seed + patient_id)

    try:
        from faker import Faker

        fake = Faker("pt_BR")
        nome = fake.name()
        idade = random.randint(18, 95)
    except ImportError:
        logger.warning("Faker não instalado. Usando nomes genéricos.")
        nome = f"Paciente Sintético {patient_id:03d}"
        idade = random.randint(18, 95)

    # Gerar diagnósticos (1 a 3)
    num_diags = random.randint(1, 3)
    diagnosticos = random.sample(DIAGNOSTICOS, num_diags)

    # Gerar medicamentos em uso (0 a 5)
    num_meds = random.randint(0, 5)
    medicamentos = random.sample(MEDICAMENTOS, min(num_meds, len(MEDICAMENTOS)))

    # Gerar exames (2 a 6)
    num_exames = random.randint(2, 6)
    exames_selecionados = random.sample(EXAMES, num_exames)
    exames = []
    now = datetime.now()
    for exame in exames_selecionados:
        status = random.choice(["pendente", "realizado", "resultado_disponivel"])
        data = now - timedelta(days=random.randint(0, 30))
        exames.append({
            **exame,
            "status": status,
            "data_solicitacao": data.strftime("%Y-%m-%d"),
            "resultado": _gerar_resultado(exame["nome"]) if status == "resultado_disponivel" else None,
        })

    # Alergias
    num_alergias = random.choice([0, 0, 0, 1, 1, 2])
    if num_alergias == 0:
        alergias = ["Nenhuma alergia conhecida"]
    else:
        alergias = random.sample(
            [a for a in ALERGIAS if a != "Nenhuma alergia conhecida"],
            min(num_alergias, len(ALERGIAS) - 1),
        )

    # Sinais vitais
    sinais_vitais = {
        "pa_sistolica": random.randint(100, 180),
        "pa_diastolica": random.randint(60, 110),
        "fc": random.randint(55, 120),
        "fr": random.randint(12, 28),
        "temperatura": round(random.uniform(35.5, 39.5), 1),
        "spo2": random.randint(88, 100),
    }

    return {
        "id": f"PAC-{patient_id:04d}",
        "nome_anonimizado": f"[PACIENTE_{patient_id:04d}]",
        "nome_ficticio": nome,
        "idade": idade,
        "sexo": random.choice(["M", "F"]),
        "setor": random.choice(SETORES),
        "diagnosticos": diagnosticos,
        "medicamentos_em_uso": medicamentos,
        "exames": exames,
        "alergias": alergias,
        "sinais_vitais": sinais_vitais,
        "queixa_principal": _gerar_queixa(diagnosticos),
        "data_admissao": (now - timedelta(days=random.randint(0, 15))).strftime("%Y-%m-%d"),
    }


def _gerar_resultado(exame_nome: str) -> str:
    """Gera resultado fictício para um exame."""
    resultados = {
        "Hemograma Completo": f"Hb: {random.uniform(8.0, 16.0):.1f} g/dL, "
        f"Ht: {random.uniform(25.0, 50.0):.1f}%, "
        f"Leuc: {random.randint(3000, 18000)}/mm³, "
        f"Plaq: {random.randint(100000, 400000)}/mm³",
        "Glicemia de Jejum": f"{random.randint(70, 250)} mg/dL",
        "HbA1c": f"{random.uniform(4.5, 12.0):.1f}%",
        "Creatinina Sérica": f"{random.uniform(0.5, 5.0):.2f} mg/dL",
        "Ureia": f"{random.randint(15, 120)} mg/dL",
        "TSH": f"{random.uniform(0.2, 15.0):.2f} mUI/L",
        "Potássio Sérico": f"{random.uniform(2.8, 6.5):.1f} mEq/L",
        "PCR": f"{random.uniform(0.1, 100.0):.1f} mg/L",
        "INR": f"{random.uniform(0.8, 4.5):.2f}",
    }
    return resultados.get(exame_nome, "Resultado dentro dos parâmetros normais")


def _gerar_queixa(diagnosticos: list[str]) -> str:
    """Gera queixa principal baseada nos diagnósticos."""
    queixas = {
        "Hipertensão Arterial Sistêmica": "Cefaleia e tontura há 3 dias",
        "Diabetes Mellitus Tipo 2": "Polidipsia e poliúria há 1 semana",
        "Pneumonia Adquirida na Comunidade": "Tosse produtiva e febre há 5 dias",
        "Insuficiência Cardíaca Congestiva": "Dispneia aos esforços e edema de MMII",
        "Infecção do Trato Urinário": "Disúria e polaciúria há 3 dias",
        "Doença Pulmonar Obstrutiva Crônica": "Dispneia progressiva e sibilos",
        "Asma Brônquica": "Crises de broncoespasmo e tosse noturna",
        "Fibrilação Atrial": "Palpitações e desconforto torácico",
        "Insuficiência Renal Crônica": "Edema generalizado e oligúria",
        "Trombose Venosa Profunda": "Dor e edema em membro inferior esquerdo",
        "Anemia Ferropriva": "Fraqueza, palidez e cansaço progressivo",
        "Hipotireoidismo": "Ganho de peso, fadiga e constipação",
        "Gastrite Crônica": "Epigastralgia e queimação retroesternal",
        "Artrite Reumatóide": "Dor articular e rigidez matinal",
        "Depressão Maior": "Tristeza persistente e insônia há 2 meses",
    }
    if diagnosticos:
        return queixas.get(diagnosticos[0], "Mal-estar geral")
    return "Checkup de rotina"


def generate_synthetic_dataset(
    num_patients: int = 20,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Gera um dataset completo de pacientes sintéticos.

    Args:
        num_patients: Número de pacientes a gerar
        output_path: Caminho para salvar (JSON)
        seed: Semente para reprodutibilidade
    """
    patients = []
    for i in range(1, num_patients + 1):
        patient = generate_synthetic_patient(i, seed=seed)
        patients.append(patient)

    logger.info("Gerados %d pacientes sintéticos", len(patients))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(patients, f, ensure_ascii=False, indent=2)
        logger.info("Dataset sintético salvo em: %s", output_path)

    return patients


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    BASE = Path(__file__).resolve().parents[3]
    generate_synthetic_dataset(
        num_patients=20,
        output_path=BASE / "data" / "synthetic" / "synthetic_patients.json",
    )
