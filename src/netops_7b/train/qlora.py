"""QLoRA training configuration and trainer for NetOps-7B.

Uses Unsloth for 2-4x faster fine-tuning with 4-bit quantization.
Base model: Qwen 2.5 7B Instruct.

Hyperparameters:
  - LoRA rank: 16
  - LoRA alpha: 32
  - LoRA dropout: 0.05
  - Target modules: all linear layers
  - Quantization: 4-bit NormalFloat (NF4) with double quantization
  - Epochs: 3
  - Learning rate: 2e-4
  - Batch size: 2
  - Gradient accumulation steps: 4
  - Effective batch size: 8
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""

    # Model
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 4096
    dtype: str | None = None  # auto-detect; float16 for V100, bfloat16 for A100+
    load_in_4bit: bool = True

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"  # Unsloth's optimized checkpointing
    use_rslora: bool = False

    # Training
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 42

    # Data
    dataset_path: str = "data/train.jsonl"
    dataset_format: str = "sharegpt"  # 'alpaca' or 'sharegpt'

    # Output
    output_dir: str = "outputs/netops-7b-qlora"
    hub_model_id: str = "cwccie/netops-7b"

    # Wandb
    report_to: str = "wandb"
    run_name: str = "netops-7b-qlora"
    wandb_project: str = "netops-7b"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dict."""
        return {
            k: v if not isinstance(v, list) else v
            for k, v in self.__dict__.items()
        }

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "QLoRAConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class QLoRATrainer:
    """QLoRA fine-tuning trainer using Unsloth + TRL.

    Usage:
        config = QLoRAConfig()
        trainer = QLoRATrainer(config)
        trainer.setup()
        trainer.train()
        trainer.save()
    """

    def __init__(self, config: QLoRAConfig | None = None):
        self.config = config or QLoRAConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._is_setup = False

    def setup(self) -> None:
        """Load model, apply LoRA adapters, prepare dataset and trainer."""
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset

        # Load base model with 4-bit quantization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )

        # Apply LoRA adapters to all linear layers
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            use_rslora=self.config.use_rslora,
        )

        # Load and format dataset
        dataset = load_dataset("json", data_files=self.config.dataset_path, split="train")

        # Build chat template function
        def format_sharegpt(example):
            """Format ShareGPT conversation for the model's chat template."""
            messages = []
            for turn in example.get("conversations", []):
                role_map = {"system": "system", "human": "user", "gpt": "assistant"}
                role = role_map.get(turn["from"], turn["from"])
                messages.append({"role": role, "content": turn["value"]})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": text}

        def format_alpaca(example):
            """Format Alpaca example for the model's chat template."""
            messages = []
            if example.get("system"):
                messages.append({"role": "system", "content": example["system"]})
            instruction = example["instruction"]
            if example.get("input"):
                instruction += f"\n\n{example['input']}"
            messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": example["output"]})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            return {"text": text}

        if self.config.dataset_format == "sharegpt":
            dataset = dataset.map(format_sharegpt)
        else:
            dataset = dataset.map(format_alpaca)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
        )

        # SFT Trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=True,  # Pack multiple examples into one sequence for efficiency
        )

        self._is_setup = True

    def train(self) -> dict[str, Any]:
        """Run training. Returns training metrics."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before train()")

        result = self.trainer.train()
        return {
            "train_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime"),
            "train_samples_per_second": result.metrics.get("train_samples_per_second"),
            "epochs": self.config.num_train_epochs,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save the fine-tuned LoRA adapters."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before save()")

        save_path = Path(path or self.config.output_dir) / "final"
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        # Save training config alongside
        self.config.save(save_path / "training_config.json")

        return save_path

    def push_to_hub(self, repo_id: str | None = None) -> None:
        """Push fine-tuned model to Hugging Face Hub."""
        if not self._is_setup:
            raise RuntimeError("Call setup() before push_to_hub()")

        repo_id = repo_id or self.config.hub_model_id
        self.model.push_to_hub(repo_id)
        self.tokenizer.push_to_hub(repo_id)

    def print_trainable_params(self) -> dict[str, int]:
        """Print and return trainable parameter count."""
        if self.model is None:
            raise RuntimeError("Call setup() first")

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_percent": round(100 * trainable / total, 2),
        }
