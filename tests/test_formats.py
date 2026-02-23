"""Tests for format converters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from netops_7b.data.curate import InstructionPair
from netops_7b.data.formats import (
    AlpacaFormatter,
    ShareGPTFormatter,
    convert_file,
)


@pytest.fixture
def sample_pairs() -> list[InstructionPair]:
    return [
        InstructionPair(
            instruction="Configure OSPF area 0",
            response="router ospf 1\n network 10.0.0.0 0.0.0.255 area 0",
            category="config_generation",
            vendor="cisco",
            complexity="basic",
        ),
        InstructionPair(
            instruction="Explain show ip bgp summary",
            response="The show ip bgp summary command displays BGP neighbor status.",
            category="command_explanation",
            vendor="cisco",
            complexity="intermediate",
        ),
    ]


class TestAlpacaFormatter:
    def test_format_one(self, sample_pairs):
        formatter = AlpacaFormatter()
        result = formatter.format_one(sample_pairs[0])
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert result["instruction"] == sample_pairs[0].instruction
        assert result["output"] == sample_pairs[0].response
        assert result["input"] == ""

    def test_format_one_with_system(self, sample_pairs):
        formatter = AlpacaFormatter(include_system=True)
        result = formatter.format_one(sample_pairs[0])
        assert "system" in result
        assert "NetOps-7B" in result["system"]

    def test_format_one_without_system(self, sample_pairs):
        formatter = AlpacaFormatter(include_system=False)
        result = formatter.format_one(sample_pairs[0])
        assert "system" not in result

    def test_format_batch(self, sample_pairs):
        formatter = AlpacaFormatter()
        results = formatter.format_batch(sample_pairs)
        assert len(results) == len(sample_pairs)

    def test_save_json(self, sample_pairs):
        formatter = AlpacaFormatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = formatter.save(sample_pairs, Path(tmpdir) / "alpaca.json")
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2

    def test_save_jsonl(self, sample_pairs):
        formatter = AlpacaFormatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = formatter.save_jsonl(sample_pairs, Path(tmpdir) / "alpaca.jsonl")
            assert path.exists()
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

    def test_custom_system_prompt(self, sample_pairs):
        formatter = AlpacaFormatter(system_prompt="Custom prompt", include_system=True)
        result = formatter.format_one(sample_pairs[0])
        assert result["system"] == "Custom prompt"


class TestShareGPTFormatter:
    def test_format_one(self, sample_pairs):
        formatter = ShareGPTFormatter()
        result = formatter.format_one(sample_pairs[0])
        assert "conversations" in result
        convos = result["conversations"]
        assert len(convos) == 3  # system + human + gpt
        assert convos[0]["from"] == "system"
        assert convos[1]["from"] == "human"
        assert convos[2]["from"] == "gpt"

    def test_format_one_without_system(self, sample_pairs):
        formatter = ShareGPTFormatter(include_system=False)
        result = formatter.format_one(sample_pairs[0])
        convos = result["conversations"]
        assert len(convos) == 2
        assert convos[0]["from"] == "human"

    def test_format_multi_turn(self, sample_pairs):
        formatter = ShareGPTFormatter()
        result = formatter.format_multi_turn(sample_pairs)
        convos = result["conversations"]
        # system + 2*(human + gpt) = 5
        assert len(convos) == 5
        assert convos[0]["from"] == "system"
        assert convos[1]["from"] == "human"
        assert convos[2]["from"] == "gpt"
        assert convos[3]["from"] == "human"
        assert convos[4]["from"] == "gpt"

    def test_format_batch(self, sample_pairs):
        formatter = ShareGPTFormatter()
        results = formatter.format_batch(sample_pairs)
        assert len(results) == 2

    def test_save_json(self, sample_pairs):
        formatter = ShareGPTFormatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = formatter.save(sample_pairs, Path(tmpdir) / "sharegpt.json")
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 2

    def test_save_jsonl(self, sample_pairs):
        formatter = ShareGPTFormatter()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = formatter.save_jsonl(sample_pairs, Path(tmpdir) / "sharegpt.jsonl")
            assert path.exists()


class TestConvertFile:
    def test_convert_to_alpaca(self, sample_pairs):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write input JSONL
            input_path = Path(tmpdir) / "input.jsonl"
            with open(input_path, "w") as f:
                for pair in sample_pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")

            output_path = Path(tmpdir) / "output.json"
            result = convert_file(input_path, output_path, output_format="alpaca")
            assert result.exists()

            with open(result) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "instruction" in data[0]

    def test_convert_to_sharegpt(self, sample_pairs):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            with open(input_path, "w") as f:
                for pair in sample_pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")

            output_path = Path(tmpdir) / "output.json"
            result = convert_file(input_path, output_path, output_format="sharegpt")
            assert result.exists()

            with open(result) as f:
                data = json.load(f)
            assert len(data) == 2
            assert "conversations" in data[0]

    def test_convert_invalid_format(self, sample_pairs):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            with open(input_path, "w") as f:
                for pair in sample_pairs:
                    f.write(json.dumps(pair.to_dict()) + "\n")

            with pytest.raises(ValueError, match="Unsupported format"):
                convert_file(input_path, Path(tmpdir) / "out.json", output_format="invalid")
