"""Tests for evaluation modules (mocked — no model loading)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from netops_7b.eval.harness import EvaluationHarness, EvalResult, EvalSummary
from netops_7b.eval.benchmark import BenchmarkRunner, BenchmarkResult


class TestEvalResult:
    def test_creation(self):
        result = EvalResult(
            example_id="eval_0001",
            category="config_generation",
            instruction="Configure OSPF",
            reference="router ospf 1",
            prediction="router ospf 1\n network 10.0.0.0",
            scores={"syntax_score": 0.85},
        )
        assert result.example_id == "eval_0001"
        assert result.scores["syntax_score"] == 0.85


class TestEvaluationHarness:
    def test_init(self):
        harness = EvaluationHarness(model_path="test/model")
        assert harness.model_path == "test/model"
        assert harness.model is None

    def test_generate_without_model_raises(self):
        harness = EvaluationHarness()
        with pytest.raises(RuntimeError, match="Call load_model"):
            harness.generate("test instruction")

    def test_load_test_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test.jsonl"
            with open(test_path, "w") as f:
                f.write(json.dumps({
                    "instruction": "Configure OSPF",
                    "response": "router ospf 1",
                    "category": "config_generation",
                }) + "\n")
                f.write(json.dumps({
                    "instruction": "Explain show ip bgp",
                    "response": "Shows BGP summary",
                    "category": "command_explanation",
                }) + "\n")

            harness = EvaluationHarness()
            examples = harness.load_test_set(test_path)
            assert len(examples) == 2
            assert examples[0]["instruction"] == "Configure OSPF"

    def test_save_results(self):
        harness = EvaluationHarness()
        harness.results = [
            EvalResult("eval_0001", "config_generation", "inst", "ref", "pred",
                       scores={"score": 0.9}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = harness.save_results(Path(tmpdir) / "results.json")
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert len(data["results"]) == 1

    def test_generate_test_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = EvaluationHarness.generate_test_set(
                Path(tmpdir) / "test.jsonl", size=50
            )
            assert path.exists()

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) > 0


class TestBenchmarkRunner:
    def test_init(self):
        runner = BenchmarkRunner(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            finetuned_model="test/model",
        )
        assert runner.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert runner.finetuned_model == "test/model"

    def test_save_and_load(self):
        base_summary = EvalSummary(
            total_examples=100,
            category_scores={},
            overall_scores={"rouge1": 0.4, "syntax": 0.5},
            eval_time_seconds=60.0,
            model_name="base",
        )
        ft_summary = EvalSummary(
            total_examples=100,
            category_scores={},
            overall_scores={"rouge1": 0.6, "syntax": 0.8},
            eval_time_seconds=60.0,
            model_name="finetuned",
        )
        result = BenchmarkResult(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_summary,
            finetuned_scores=ft_summary,
            improvements={"rouge1": 0.2, "syntax": 0.3},
            test_set_size=100,
            benchmark_time_seconds=120.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner()
            path = runner.save(result, Path(tmpdir) / "bench.json")
            assert path.exists()

            loaded = BenchmarkRunner.load(path)
            assert loaded["test_set_size"] == 100
            assert loaded["improvements"]["rouge1"] == 0.2

    def test_markdown_table(self):
        base_summary = EvalSummary(
            total_examples=10,
            category_scores={},
            overall_scores={"rouge1": 0.4},
            eval_time_seconds=1.0,
            model_name="base",
        )
        ft_summary = EvalSummary(
            total_examples=10,
            category_scores={},
            overall_scores={"rouge1": 0.6},
            eval_time_seconds=1.0,
            model_name="ft",
        )
        result = BenchmarkResult(
            base_model="base",
            finetuned_model="ft",
            base_scores=base_summary,
            finetuned_scores=ft_summary,
            improvements={"rouge1": 0.2},
            test_set_size=10,
            benchmark_time_seconds=2.0,
        )

        table = result.to_markdown_table()
        assert "| Metric |" in table
        assert "rouge1" in table
        assert "+0.2000" in table
