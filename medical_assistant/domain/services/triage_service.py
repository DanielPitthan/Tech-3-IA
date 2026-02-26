"""
Serviços de domínio: Triagem, Exames e Tratamento.
"""

from __future__ import annotations

from medical_assistant.domain.entities.patient import Patient
from medical_assistant.domain.value_objects.triage_level import TriageLevel


class TriageService:
    """Serviço de triagem baseado em sinais vitais e queixa."""

    CRITICAL_KEYWORDS = [
        "parada", "pcr", "anafilaxia", "choque", "convulsão",
        "hemorragia", "iam", "avc", "politrauma", "rebaixamento",
    ]

    URGENT_KEYWORDS = [
        "dispneia", "dor torácica", "febre alta", "sangramento",
        "fratura", "cefaleia súbita", "síncope", "taquicardia",
    ]

    def classify(self, patient: Patient) -> TriageLevel:
        """Classifica nível de triagem do paciente."""
        # Sinais vitais críticos
        if patient.has_critical_vitals:
            return TriageLevel.CRITICO

        # Palavras-chave na queixa
        queixa = patient.queixa_principal.lower()
        for kw in self.CRITICAL_KEYWORDS:
            if kw in queixa:
                return TriageLevel.CRITICO

        for kw in self.URGENT_KEYWORDS:
            if kw in queixa:
                return TriageLevel.URGENTE

        # Setor UTI = urgente por padrão
        if patient.setor and "uti" in patient.setor.lower():
            return TriageLevel.URGENTE

        return TriageLevel.REGULAR


class ExamService:
    """Serviço de verificação de exames."""

    def get_pending_exams(self, patient: Patient) -> list[dict]:
        """Retorna exames pendentes do paciente."""
        return patient.pending_exams

    def get_available_results(self, patient: Patient) -> list[dict]:
        """Retorna exames com resultados disponíveis."""
        return patient.available_results

    def suggest_exams(self, patient: Patient) -> list[str]:
        """Sugere exames baseado nos diagnósticos."""
        suggestions: list[str] = []
        existing = {e["nome"] for e in patient.exames}

        diag_exams = {
            "Diabetes Mellitus Tipo 2": ["HbA1c", "Glicemia de Jejum", "Creatinina Sérica"],
            "Hipertensão Arterial Sistêmica": ["Creatinina Sérica", "Potássio Sérico", "ECG"],
            "Pneumonia Adquirida na Comunidade": ["Hemograma Completo", "PCR", "Raio-X de Tórax"],
            "Insuficiência Cardíaca Congestiva": ["ECG", "Ecocardiograma", "Hemograma Completo"],
            "Insuficiência Renal Crônica": ["Creatinina Sérica", "Ureia", "Potássio Sérico"],
            "Fibrilação Atrial": ["ECG", "INR", "Ecocardiograma"],
            "Hipotireoidismo": ["TSH", "T4 Livre"],
            "Trombose Venosa Profunda": ["INR", "Hemograma Completo"],
        }

        for diag in patient.diagnosticos:
            recommended = diag_exams.get(diag, [])
            for exam in recommended:
                if exam not in existing and exam not in suggestions:
                    suggestions.append(exam)

        return suggestions


class TreatmentService:
    """Serviço de sugestão de tratamento."""

    def check_drug_interactions(
        self, patient: Patient, suggested_drug: str
    ) -> list[str]:
        """Verifica interações medicamentosas potenciais."""
        interactions: list[str] = []

        known_interactions = {
            ("Warfarina", "AAS"): "Risco aumentado de sangramento",
            ("Warfarina", "Amoxicilina"): "Pode potencializar efeito anticoagulante",
            ("Enalapril", "Losartana"): "Duplo bloqueio SRAA — risco de hipercalemia",
            ("Metformina", "contraste"): "Risco de acidose láctica com contraste iodado",
            ("Insulina NPH", "Prednisona"): "Corticoides aumentam glicemia",
        }

        current_meds = [m["nome"] for m in patient.medicamentos_em_uso]

        for (drug_a, drug_b), risk in known_interactions.items():
            if (suggested_drug == drug_a and drug_b in current_meds) or (
                suggested_drug == drug_b and drug_a in current_meds
            ):
                interactions.append(f"⚠️ {drug_a} + {drug_b}: {risk}")

        return interactions

    def check_allergies(self, patient: Patient, suggested_drug: str) -> list[str]:
        """Verifica alergias do paciente ao medicamento sugerido."""
        alerts: list[str] = []

        allergy_groups = {
            "Penicilina": ["Amoxicilina", "Ampicilina", "Penicilina"],
            "Cefalosporina": ["Cefalexina", "Ceftriaxona"],
            "Sulfa": ["Sulfametoxazol"],
            "AINEs": ["Ibuprofeno", "Diclofenaco", "Naproxeno"],
        }

        for allergy in patient.alergias:
            related_drugs = allergy_groups.get(allergy, [allergy])
            if suggested_drug in related_drugs:
                alerts.append(
                    f"🚨 ALERGIA: Paciente é alérgico a {allergy}. "
                    f"{suggested_drug} é contraindicado!"
                )

        return alerts
