"""
Configurações do modelo LLM para inferência e fine-tuning.

Centraliza todos os parâmetros de modelo e geração,
carregando de configs/model_config.yaml e configs/qlora_config.yaml.
Suporta Ollama (Llama 3) e HuggingFace (Llama 3.1-8B-Instruct).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Carrega arquivo YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class QuantizationConfig:
    """Configuração de quantização BitsAndBytes."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraConfig:
    """Configuração LoRA/QLoRA."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainingConfig:
    """Configuração de treinamento."""

    output_dir: str = "./models/llama3-medical"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    report_to: str = "none"
    seed: int = 42
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(
        default_factory=lambda: {"use_reentrant": False}
    )


@dataclass
class GenerationConfig:
    """Configuração de geração (inferência)."""

    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    do_sample: bool = True


@dataclass
class SFTConfig:
    """Configuração do SFTTrainer."""

    max_seq_length: int = 512
    packing: bool = False
    dataset_text_field: str = "text"


@dataclass
class ModelConfig:
    """Configuração completa do modelo."""

    # Modelo base
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    revision: str = "main"
    trust_remote_code: bool = False
    torch_dtype: str = "float16"

    # Sub-configurações
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)

    @classmethod
    def from_yaml(
        cls,
        model_config_path: str | Path | None = None,
        qlora_config_path: str | Path | None = None,
    ) -> ModelConfig:
        """
        Carrega configuração a partir de arquivos YAML.

        Args:
            model_config_path: Caminho para model_config.yaml
            qlora_config_path: Caminho para qlora_config.yaml
        """
        model_path = Path(model_config_path) if model_config_path else CONFIGS_DIR / "model_config.yaml"
        qlora_path = Path(qlora_config_path) if qlora_config_path else CONFIGS_DIR / "qlora_config.yaml"

        config = cls()

        if model_path.exists():
            model_cfg = _load_yaml(model_path)
            # Modelo base
            model_section = model_cfg.get("model", {})
            config.model_name = model_section.get("name", config.model_name)
            config.provider = model_section.get("provider", config.provider)
            config.base_url = model_section.get("base_url", config.base_url)
            config.revision = model_section.get("revision", config.revision)
            config.trust_remote_code = model_section.get("trust_remote_code", config.trust_remote_code)
            config.torch_dtype = model_section.get("torch_dtype", config.torch_dtype)

            # Quantização
            quant_section = model_cfg.get("quantization", {})
            config.quantization = QuantizationConfig(**{
                k: quant_section.get(k, getattr(config.quantization, k))
                for k in QuantizationConfig.__dataclass_fields__
            })

            # Geração
            gen_section = model_cfg.get("generation", {})
            config.generation = GenerationConfig(**{
                k: gen_section.get(k, getattr(config.generation, k))
                for k in GenerationConfig.__dataclass_fields__
            })

            logger.info("Model config carregado de: %s", model_path)

        if qlora_path.exists():
            qlora_cfg = _load_yaml(qlora_path)

            # LoRA
            lora_section = qlora_cfg.get("lora", {})
            config.lora = LoraConfig(**{
                k: lora_section.get(k, getattr(config.lora, k))
                for k in LoraConfig.__dataclass_fields__
            })

            # Training
            train_section = qlora_cfg.get("training", {})
            config.training = TrainingConfig(**{
                k: train_section.get(k, getattr(config.training, k))
                for k in TrainingConfig.__dataclass_fields__
            })

            # SFT
            sft_section = qlora_cfg.get("sft", {})
            config.sft = SFTConfig(**{
                k: sft_section.get(k, getattr(config.sft, k))
                for k in SFTConfig.__dataclass_fields__
            })

            logger.info("QLoRA config carregado de: %s", qlora_path)

        return config

    def get_bnb_config_dict(self) -> dict[str, Any]:
        """Retorna dicionário para BitsAndBytesConfig."""
        import torch

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        compute_dtype = dtype_map.get(self.quantization.bnb_4bit_compute_dtype, torch.float16)

        return {
            "load_in_4bit": self.quantization.load_in_4bit,
            "bnb_4bit_quant_type": self.quantization.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": self.quantization.bnb_4bit_use_double_quant,
        }

    def get_lora_config_dict(self) -> dict[str, Any]:
        """Retorna dicionário para peft.LoraConfig."""
        return {
            "r": self.lora.r,
            "lora_alpha": self.lora.lora_alpha,
            "lora_dropout": self.lora.lora_dropout,
            "bias": self.lora.bias,
            "task_type": self.lora.task_type,
            "target_modules": self.lora.target_modules,
        }

    def get_training_args_dict(self) -> dict[str, Any]:
        """Retorna dicionário para TrainingArguments."""
        return {
            k: getattr(self.training, k)
            for k in TrainingConfig.__dataclass_fields__
        }
