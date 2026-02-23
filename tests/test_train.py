"""Tests for training configuration (mocked — no GPU required)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from netops_7b.train.qlora import QLoRAConfig, QLoRATrainer
from netops_7b.train.export import GGUFExporter, SUPPORTED_QUANTS


class TestQLoRAConfig:
    def test_default_values(self):
        config = QLoRAConfig()
        assert config.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-4
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 4
        assert config.load_in_4bit is True

    def test_effective_batch_size(self):
        config = QLoRAConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 8

    def test_custom_values(self):
        config = QLoRAConfig(
            lora_rank=32,
            learning_rate=1e-4,
            num_train_epochs=5,
        )
        assert config.lora_rank == 32
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 5

    def test_to_dict(self):
        config = QLoRAConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["base_model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert d["lora_rank"] == 16
        assert isinstance(d["target_modules"], list)

    def test_save_and_load(self):
        config = QLoRAConfig(lora_rank=32, learning_rate=1e-5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(path)
            assert path.exists()

            loaded = QLoRAConfig.load(path)
            assert loaded.lora_rank == 32
            assert loaded.learning_rate == 1e-5

    def test_target_modules_all_linear(self):
        config = QLoRAConfig()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
        assert config.target_modules == expected


class TestQLoRATrainer:
    def test_init_default_config(self):
        trainer = QLoRATrainer()
        assert trainer.config.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert trainer.model is None
        assert trainer._is_setup is False

    def test_init_custom_config(self):
        config = QLoRAConfig(lora_rank=32)
        trainer = QLoRATrainer(config=config)
        assert trainer.config.lora_rank == 32

    def test_train_without_setup_raises(self):
        trainer = QLoRATrainer()
        with pytest.raises(RuntimeError, match="Call setup"):
            trainer.train()

    def test_save_without_setup_raises(self):
        trainer = QLoRATrainer()
        with pytest.raises(RuntimeError, match="Call setup"):
            trainer.save()

    def test_push_without_setup_raises(self):
        trainer = QLoRATrainer()
        with pytest.raises(RuntimeError, match="Call setup"):
            trainer.push_to_hub()


class TestGGUFExporter:
    def test_init_defaults(self):
        exporter = GGUFExporter()
        assert exporter.model_path == "outputs/netops-7b-qlora/final"
        assert exporter.output_dir == "outputs/gguf"
        assert exporter.model_name == "netops-7b"

    def test_supported_quants(self):
        assert "q4_k_m" in SUPPORTED_QUANTS
        assert "q5_k_m" in SUPPORTED_QUANTS
        assert "q8_0" in SUPPORTED_QUANTS

    def test_create_ollama_modelfile(self):
        exporter = GGUFExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_path = Path(tmpdir) / "model.gguf"
            gguf_path.touch()

            mf_path = exporter.create_ollama_modelfile(gguf_path)
            assert mf_path.exists()

            content = mf_path.read_text()
            assert "FROM model.gguf" in content
            assert "PARAMETER temperature" in content
            assert "SYSTEM" in content
            assert "NetOps-7B" in content

    def test_get_file_sizes_empty(self):
        exporter = GGUFExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = tmpdir
            sizes = exporter.get_file_sizes()
            assert sizes == {}

    def test_get_file_sizes_with_files(self):
        exporter = GGUFExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = tmpdir
            # Create a small fake GGUF file
            fake = Path(tmpdir) / "model-q4_k_m.gguf"
            fake.write_bytes(b"x" * 1024 * 1024)  # 1 MB

            sizes = exporter.get_file_sizes()
            assert "model-q4_k_m.gguf" in sizes
            assert "MB" in sizes["model-q4_k_m.gguf"]
