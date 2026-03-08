"""
Pipeline de fine-tuning do Llama 3.1-8B-Instruct com QLoRA.

Utiliza BitsAndBytes para quantização 4-bit (NF4),
PEFT/LoRA para adaptação eficiente de parâmetros,
e TRL SFTTrainer para Supervised Fine-Tuning.

Hardware alvo: GPU local com 8-12GB VRAM (ex: RTX 3060/4060).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from medical_assistant.infrastructure.llm.model_config import ModelConfig

logger = logging.getLogger(__name__)


class Llama3QLoRATrainer:
    """
    Pipeline completo de fine-tuning do Llama 3.1-8B-Instruct com QLoRA.

    Etapas:
    1. Carrega modelo com quantização 4-bit (BitsAndBytes)
    2. Prepara modelo para treinamento k-bit
    3. Aplica LoRA com PEFT
    4. Treina com SFTTrainer (TRL)
    5. Salva apenas o adapter LoRA (~50-100MB)
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Inicializa o trainer.

        Args:
            config: Configuração do modelo. Se None, carrega dos YAML.
        """
        self.config = config or ModelConfig.from_yaml()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self) -> None:
        """
        Carrega o modelo base com quantização 4-bit e o tokenizer.

        O modelo Llama 3.1-8B-Instruct ocupa ~5GB em 4-bit NF4,
        permitindo fine-tuning em GPUs com 8-12GB VRAM.
        """
        logger.info("Carregando modelo: %s", self.config.model_name)
        logger.info("Quantização: 4-bit NF4 com double quantization")

        # Configuração BitsAndBytes
        bnb_config = BitsAndBytesConfig(**self.config.get_bnb_config_dict())

        # Carregar tokenizer
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
            token=hf_token,
        )

        # Definir pad token — Llama 3.1 possui token dedicado para padding
        if self.tokenizer.pad_token is None:
            if "<|finetune_right_pad_id|>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<|finetune_right_pad_id|>"
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

        # Carregar modelo quantizado
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
            token=hf_token,
        )

        # Sincronizar pad_token_id com model config e generation config
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Desabilitar cache para treinamento
        self.model.config.use_cache = False

        # Preparar para treinamento k-bit (com use_reentrant=False para PyTorch >=2.9)
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        logger.info("Modelo carregado com sucesso. Parâmetros totais: %s", _count_params(self.model))

    def apply_lora(self) -> None:
        """Aplica adaptadores LoRA ao modelo."""
        if self.model is None:
            raise RuntimeError("Modelo não carregado. Chame load_model_and_tokenizer() primeiro.")

        lora_config = LoraConfig(**self.config.get_lora_config_dict())

        self.model = get_peft_model(self.model, lora_config)

        trainable, total = _count_trainable_params(self.model)
        logger.info(
            "LoRA aplicado. Parâmetros treináveis: %s / %s (%.2f%%)",
            _format_num(trainable),
            _format_num(total),
            100 * trainable / total,
        )

    def load_dataset(
        self,
        train_path: str | Path,
        val_path: str | Path | None = None,
    ) -> tuple[Dataset, Dataset | None]:
        """
        Carrega dataset de treinamento e validação a partir de JSONL.

        Args:
            train_path: Caminho para o JSONL de treinamento
            val_path: Caminho para o JSONL de validação (opcional)

        Returns:
            Tupla (train_dataset, val_dataset)
        """
        train_data = _load_jsonl(train_path)
        train_dataset = Dataset.from_list(train_data)
        logger.info("Dataset de treino carregado: %d amostras", len(train_dataset))

        val_dataset = None
        if val_path:
            val_data = _load_jsonl(val_path)
            val_dataset = Dataset.from_list(val_data)
            logger.info("Dataset de validação carregado: %d amostras", len(val_dataset))

        return train_dataset, val_dataset

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
    ) -> None:
        """
        Executa o fine-tuning com SFTTrainer.

        Args:
            train_dataset: Dataset de treinamento (com campo 'text')
            val_dataset: Dataset de validação (opcional)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modelo/tokenizer não carregados.")

        # Training Arguments (SFTConfig extends TrainingArguments with SFT params)
        training_args_dict = self.config.get_training_args_dict()
        training_args_dict["max_seq_length"] = self.config.sft.max_seq_length
        training_args_dict["packing"] = self.config.sft.packing
        training_args_dict["dataset_text_field"] = self.config.sft.dataset_text_field
        training_args = SFTConfig(**training_args_dict)

        # Remover colunas extras que não são usadas pelo SFTTrainer
        # (ex: 'label' é string e causa erro ao tentar criar tensores)
        text_field = self.config.sft.dataset_text_field
        keep_cols = {text_field}
        train_dataset = train_dataset.select_columns(
            [c for c in train_dataset.column_names if c in keep_cols]
        )
        if val_dataset is not None:
            val_dataset = val_dataset.select_columns(
                [c for c in val_dataset.column_names if c in keep_cols]
            )

        # SFT Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
        )

        logger.info("Iniciando fine-tuning...")
        logger.info("  Épocas: %d", self.config.training.num_train_epochs)
        logger.info("  Batch size efetivo: %d", (
            self.config.training.per_device_train_batch_size
            * self.config.training.gradient_accumulation_steps
        ))
        logger.info("  Learning rate: %s", self.config.training.learning_rate)
        logger.info("  Output dir: %s", self.config.training.output_dir)

        # Treinar
        train_result = self.trainer.train()

        # Log de métricas
        metrics = train_result.metrics
        logger.info("Fine-tuning concluído!")
        logger.info("  Loss final: %.4f", metrics.get("train_loss", 0.0))
        logger.info("  Runtime: %.1f segundos", metrics.get("train_runtime", 0.0))
        logger.info(
            "  Samples/segundo: %.2f",
            metrics.get("train_samples_per_second", 0.0),
        )

    def save_model(self, output_dir: str | Path | None = None) -> Path:
        """
        Salva apenas o adapter LoRA (não o modelo base completo).

        O adapter LoRA tem ~50-100MB vs ~16GB do modelo completo.
        Para inferência, carrega-se o modelo base + adapter.
        """
        output_dir = Path(output_dir or self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.trainer:
            self.trainer.save_model(str(output_dir))
        elif self.model:
            self.model.save_pretrained(str(output_dir))

        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_dir))

        # Salvar configuração
        config_path = output_dir / "training_config.json"
        config_dict = {
            "model_name": self.config.model_name,
            "lora": self.config.get_lora_config_dict(),
            "training": self.config.get_training_args_dict(),
            "quantization": {
                "load_in_4bit": self.config.quantization.load_in_4bit,
                "bnb_4bit_quant_type": self.config.quantization.bnb_4bit_quant_type,
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info("Modelo salvo em: %s", output_dir)
        return output_dir

    def run_pipeline(
        self,
        train_path: str | Path,
        val_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """
        Executa o pipeline completo de fine-tuning.

        1. Carrega modelo + tokenizer com quantização 4-bit
        2. Aplica adaptadores LoRA
        3. Carrega datasets
        4. Treina com SFTTrainer
        5. Salva adapter LoRA

        Args:
            train_path: JSONL de treinamento
            val_path: JSONL de validação (opcional)
            output_dir: Diretório de saída

        Returns:
            Caminho do diretório com o modelo salvo
        """
        logger.info("=" * 60)
        logger.info("PIPELINE DE FINE-TUNING - Llama 3.1-8B QLoRA")
        logger.info("=" * 60)

        # 1. Carregar modelo
        self.load_model_and_tokenizer()

        # 2. Aplicar LoRA
        self.apply_lora()

        # 3. Carregar datasets
        train_dataset, val_dataset = self.load_dataset(train_path, val_path)

        # 4. Treinar
        self.train(train_dataset, val_dataset)

        # 5. Salvar
        saved_path = self.save_model(output_dir)

        logger.info("=" * 60)
        logger.info("PIPELINE CONCLUÍDO")
        logger.info("Modelo salvo em: %s", saved_path)
        logger.info("=" * 60)

        return saved_path


# ---------- Utilitários ----------


def _load_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """Carrega arquivo JSONL."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _count_params(model: Any) -> str:
    """Conta e formata o total de parâmetros."""
    total = sum(p.numel() for p in model.parameters())
    return _format_num(total)


def _count_trainable_params(model: Any) -> tuple[int, int]:
    """Conta parâmetros treináveis e totais."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _format_num(num: int) -> str:
    """Formata número grande (ex: 7.2B, 1.5M)."""
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    if num >= 1e6:
        return f"{num/1e6:.1f}M"
    if num >= 1e3:
        return f"{num/1e3:.1f}K"
    return str(num)
