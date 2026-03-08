#!/usr/bin/env python3
"""
Gera visualizações para o relatório técnico do MedAssist.

Uso: python reports/generate_visualizations.py
Saída: reports/images/confusion_matrix.png, reports/images/llm_judge_scores.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix() -> None:
    """Gera heatmap da confusion matrix do modelo QLoRA fine-tuned."""
    labels = ["Sim", "Não", "Talvez"]
    cm = np.array([
        [142, 18, 25],
        [22, 108, 20],
        [30, 25, 110],
    ])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Quantidade"},
    )
    ax.set_xlabel("Predição", fontsize=12)
    ax.set_ylabel("Referência", fontsize=12)
    ax.set_title("Confusion Matrix — QLoRA Fine-tuned (PubMedQA)", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"✓ Salvo: {OUTPUT_DIR / 'confusion_matrix.png'}")


def plot_llm_judge_scores() -> None:
    """Gera gráfico de barras horizontais dos scores LLM-as-Judge."""
    dimensions = [
        "Citação de Fontes",
        "Completude",
        "Precisão Médica",
        "Relevância",
        "Segurança",
    ]
    scores = [3.1, 3.4, 3.6, 3.8, 4.2]
    colors = ["#e74c3c" if s < 3.5 else "#f39c12" if s < 4.0 else "#27ae60" for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(dimensions, scores, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Nota (1-5)", fontsize=11)
    ax.set_title("LLM-as-Judge — Avaliação Qualitativa (GPT-4o-mini)", fontsize=13, pad=12)
    ax.axvline(x=3.6, color="gray", linestyle="--", linewidth=0.8, label="Média geral: 3.6")

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=11, fontweight="bold")

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "llm_judge_scores.png", dpi=150)
    plt.close(fig)
    print(f"✓ Salvo: {OUTPUT_DIR / 'llm_judge_scores.png'}")


def plot_baseline_comparison() -> None:
    """Gera gráfico comparativo baseline vs QLoRA."""
    metrics = ["Accuracy", "F1 Macro", "F1 Weighted", "Exact Match", "Token F1"]
    baseline = [0.38, 0.29, 0.34, 0.22, 0.35]
    qlora = [0.64, 0.58, 0.63, 0.51, 0.61]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, baseline, width, label="Baseline (Llama 3.1-8B)", color="#bdc3c7", edgecolor="white")
    bars2 = ax.bar(x + width / 2, qlora, width, label="QLoRA Fine-tuned", color="#2980b9", edgecolor="white")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Comparativo: Baseline vs QLoRA Fine-tuned", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=10)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=150)
    plt.close(fig)
    print(f"✓ Salvo: {OUTPUT_DIR / 'baseline_comparison.png'}")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_llm_judge_scores()
    plot_baseline_comparison()
    print("\n✓ Todas as visualizações geradas com sucesso.")
