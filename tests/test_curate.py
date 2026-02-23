"""Tests for data curation pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from netops_7b.data.curate import (
    ConfigGenerator,
    DatasetCurator,
    InstructionPair,
)


class TestInstructionPair:
    def test_creation(self):
        pair = InstructionPair(
            instruction="Configure OSPF",
            response="router ospf 1\n network 10.0.0.0 0.0.0.255 area 0",
            category="config_generation",
            vendor="cisco",
            complexity="basic",
        )
        assert pair.instruction == "Configure OSPF"
        assert pair.category == "config_generation"
        assert pair.vendor == "cisco"

    def test_uid_deterministic(self):
        pair1 = InstructionPair("a", "b", "config_generation", "cisco", "basic")
        pair2 = InstructionPair("a", "b", "config_generation", "cisco", "basic")
        assert pair1.uid == pair2.uid

    def test_uid_unique_for_different_content(self):
        pair1 = InstructionPair("a", "b", "config_generation", "cisco", "basic")
        pair2 = InstructionPair("a", "c", "config_generation", "cisco", "basic")
        assert pair1.uid != pair2.uid

    def test_to_dict(self):
        pair = InstructionPair("inst", "resp", "config_generation", "cisco", "basic")
        d = pair.to_dict()
        assert d["instruction"] == "inst"
        assert d["response"] == "resp"
        assert d["category"] == "config_generation"
        assert d["vendor"] == "cisco"
        assert "uid" in d

    def test_to_dict_with_metadata(self):
        pair = InstructionPair("inst", "resp", "config_generation", "cisco", "basic",
                               metadata={"source": "template"})
        d = pair.to_dict()
        assert d["source"] == "template"


class TestConfigGenerator:
    def test_generate_cisco(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_cisco(10)
        assert len(pairs) > 0
        assert all(p.vendor == "cisco" for p in pairs)
        assert all(p.category == "config_generation" for p in pairs)

    def test_generate_juniper(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_juniper(10)
        assert len(pairs) > 0
        assert all(p.vendor == "juniper" for p in pairs)

    def test_generate_arista(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_arista(10)
        assert len(pairs) > 0
        assert all(p.vendor == "arista" for p in pairs)

    def test_generate_troubleshooting(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_troubleshooting(10)
        assert len(pairs) > 0
        assert all(p.category == "troubleshooting" for p in pairs)

    def test_generate_explanations(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_explanations(5)
        assert len(pairs) == 5
        assert all(p.category == "command_explanation" for p in pairs)

    def test_seed_reproducibility(self):
        gen1 = ConfigGenerator(seed=123)
        gen2 = ConfigGenerator(seed=123)
        pairs1 = gen1.generate_cisco(5)
        pairs2 = gen2.generate_cisco(5)
        assert [p.uid for p in pairs1] == [p.uid for p in pairs2]

    def test_different_seeds_produce_different_data(self):
        gen1 = ConfigGenerator(seed=1)
        gen2 = ConfigGenerator(seed=2)
        pairs1 = gen1.generate_cisco(5)
        pairs2 = gen2.generate_cisco(5)
        uids1 = {p.uid for p in pairs1}
        uids2 = {p.uid for p in pairs2}
        # Very unlikely to be identical with different seeds
        assert uids1 != uids2

    def test_generated_content_not_empty(self):
        gen = ConfigGenerator(seed=42)
        pairs = gen.generate_cisco(20)
        for pair in pairs:
            assert len(pair.instruction) > 10
            assert len(pair.response) > 10


class TestDatasetCurator:
    def test_generate(self):
        curator = DatasetCurator(target_size=100, seed=42)
        pairs = curator.generate(progress=False)
        assert len(pairs) > 50  # Should get most of target

    def test_stats(self):
        curator = DatasetCurator(target_size=100, seed=42)
        pairs = curator.generate(progress=False)
        stats = curator.stats(pairs)
        assert stats["total"] == len(pairs)
        assert "categories" in stats
        assert "vendors" in stats
        assert stats["avg_length_chars"] > 0

    def test_save_jsonl(self):
        curator = DatasetCurator(target_size=50, seed=42)
        pairs = curator.generate(progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = curator.save(pairs, Path(tmpdir) / "test.jsonl", fmt="jsonl")
            assert path.exists()

            # Verify file content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == len(pairs)

            # Verify JSON structure
            data = json.loads(lines[0])
            assert "instruction" in data
            assert "response" in data

    def test_save_json(self):
        curator = DatasetCurator(target_size=50, seed=42)
        pairs = curator.generate(progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = curator.save(pairs, Path(tmpdir) / "test.json", fmt="json")
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == len(pairs)

    def test_custom_distribution(self):
        curator = DatasetCurator(
            target_size=100,
            seed=42,
            distribution={
                "cisco_config": 0.50,
                "juniper_config": 0.25,
                "arista_config": 0.25,
                "troubleshooting": 0.0,
                "explanation": 0.0,
            },
        )
        pairs = curator.generate(progress=False)
        vendors = {p.vendor for p in pairs}
        assert "cisco" in vendors
        assert "juniper" in vendors

    def test_curate_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            curator = DatasetCurator(target_size=100, seed=42)
            path = curator.curate(Path(tmpdir) / "train.jsonl", progress=False)
            assert path.exists()

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) > 0
