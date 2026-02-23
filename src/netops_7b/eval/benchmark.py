"""Benchmark comparison between NetOps-7B and base Qwen 2.5 7B.

Runs both models on identical test sets and produces side-by-side
comparison metrics across all evaluation dimensions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from netops_7b.eval.harness import EvaluationHarness, EvalSummary
from netops_7b.eval.metrics import NetworkMetrics


@dataclass
class BenchmarkResult:
    """Side-by-side benchmark comparison."""

    base_model: str
    finetuned_model: str
    base_scores: EvalSummary
    finetuned_scores: EvalSummary
    improvements: dict[str, float]
    test_set_size: int
    benchmark_time_seconds: float

    def to_markdown_table(self) -> str:
        """Generate a markdown comparison table."""
        lines = [
            "| Metric | Base (Qwen 2.5 7B) | NetOps-7B | Improvement |",
            "|--------|-------------------|-----------|-------------|",
        ]

        for metric in sorted(self.finetuned_scores.overall_scores.keys()):
            base_val = self.base_scores.overall_scores.get(metric, 0.0)
            ft_val = self.finetuned_scores.overall_scores.get(metric, 0.0)
            improvement = self.improvements.get(metric, 0.0)
            sign = "+" if improvement >= 0 else ""
            lines.append(
                f"| {metric} | {base_val:.4f} | {ft_val:.4f} | {sign}{improvement:.4f} |"
            )

        return "\n".join(lines)


class BenchmarkRunner:
    """Run comparative benchmarks between base and fine-tuned models.

    Usage:
        runner = BenchmarkRunner(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            finetuned_model="outputs/netops-7b-qlora/final",
        )
        result = runner.run("data/test.jsonl")
        print(result.to_markdown_table())
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        finetuned_model: str = "outputs/netops-7b-qlora/final",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(
        self,
        test_path: str | Path,
        system_prompt: str | None = None,
        progress: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark comparison.

        Evaluates both models on the same test set and computes improvements.

        Args:
            test_path: Path to JSONL test set.
            system_prompt: Optional system prompt.
            progress: Show progress bars.

        Returns:
            BenchmarkResult with comparison data.
        """
        start_time = time.time()

        # Evaluate base model
        print(f"\n{'='*60}")
        print(f"Evaluating base model: {self.base_model}")
        print(f"{'='*60}")

        base_harness = EvaluationHarness(
            model_path=self.base_model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        base_harness.load_model()
        base_summary = base_harness.run(test_path, system_prompt=system_prompt, progress=progress)

        # Free base model memory
        del base_harness.model
        del base_harness.tokenizer
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate fine-tuned model
        print(f"\n{'='*60}")
        print(f"Evaluating fine-tuned model: {self.finetuned_model}")
        print(f"{'='*60}")

        ft_harness = EvaluationHarness(
            model_path=self.finetuned_model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        ft_harness.load_model()
        ft_summary = ft_harness.run(test_path, system_prompt=system_prompt, progress=progress)

        # Compute improvements
        improvements: dict[str, float] = {}
        for metric in ft_summary.overall_scores:
            ft_val = ft_summary.overall_scores[metric]
            base_val = base_summary.overall_scores.get(metric, 0.0)
            improvements[metric] = ft_val - base_val

        total_time = time.time() - start_time

        # Count test set
        test_examples = base_harness.load_test_set(test_path)

        return BenchmarkResult(
            base_model=self.base_model,
            finetuned_model=self.finetuned_model,
            base_scores=base_summary,
            finetuned_scores=ft_summary,
            improvements=improvements,
            test_set_size=len(test_examples),
            benchmark_time_seconds=round(total_time, 2),
        )

    def save(self, result: BenchmarkResult, path: str | Path) -> Path:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "base_model": result.base_model,
            "finetuned_model": result.finetuned_model,
            "test_set_size": result.test_set_size,
            "benchmark_time_seconds": result.benchmark_time_seconds,
            "base_overall_scores": result.base_scores.overall_scores,
            "finetuned_overall_scores": result.finetuned_scores.overall_scores,
            "improvements": result.improvements,
            "base_category_scores": result.base_scores.category_scores,
            "finetuned_category_scores": result.finetuned_scores.category_scores,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        return path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Load saved benchmark results."""
        with open(path) as f:
            return json.load(f)
