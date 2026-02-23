"""Evaluation harness for NetOps-7B.

Runs the model against a structured test set and computes:
  - Config syntax validity (regex-based AST parsing)
  - Troubleshooting accuracy (500-question test set)
  - ROUGE and BERTScore for explanation quality
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from netops_7b.eval.metrics import NetworkMetrics


@dataclass
class EvalResult:
    """Result container for a single evaluation example."""

    example_id: str
    category: str
    instruction: str
    reference: str
    prediction: str
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""

    total_examples: int
    category_scores: dict[str, dict[str, float]]
    overall_scores: dict[str, float]
    eval_time_seconds: float
    model_name: str


class EvaluationHarness:
    """End-to-end evaluation harness for NetOps-7B.

    Usage:
        harness = EvaluationHarness(model_path="outputs/netops-7b-qlora/final")
        harness.load_model()
        summary = harness.run("data/test.jsonl")
        harness.save_results("outputs/eval_results.json")
    """

    def __init__(
        self,
        model_path: str = "outputs/netops-7b-qlora/final",
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.model = None
        self.tokenizer = None
        self.metrics = NetworkMetrics()
        self.results: list[EvalResult] = []

    def load_model(self) -> None:
        """Load the fine-tuned model for evaluation."""
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def generate(self, instruction: str, system_prompt: str | None = None) -> str:
        """Generate a response for a single instruction."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() first")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        # Decode only the generated tokens
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def load_test_set(self, path: str | Path) -> list[dict[str, Any]]:
        """Load test set from JSONL file."""
        examples = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                examples.append(data)
        return examples

    def _eval_config_example(self, prediction: str, reference: str) -> dict[str, float]:
        """Score a configuration generation example."""
        pred_score = self.metrics.score_config_syntax(prediction)
        ref_score = self.metrics.score_config_syntax(reference)

        return {
            "syntax_score": pred_score.overall_score,
            "structural_score": pred_score.structural_score,
            "completeness_score": pred_score.completeness_score,
            "reference_syntax_score": ref_score.overall_score,
        }

    def _eval_troubleshooting_example(self, prediction: str, reference: str) -> dict[str, float]:
        """Score a troubleshooting example."""
        score = self.metrics.score_troubleshooting(prediction)
        return {
            "troubleshooting_score": score.score,
            "has_systematic_approach": float(score.has_systematic_approach),
            "has_verification_commands": float(score.has_verification_commands),
            "has_root_cause_analysis": float(score.has_root_cause_analysis),
            "has_remediation_steps": float(score.has_remediation_steps),
            "layer_coverage_count": len(score.layer_coverage),
        }

    def _eval_explanation_example(self, prediction: str, reference: str) -> dict[str, float]:
        """Score a command explanation example."""
        score = self.metrics.score_explanation(prediction)
        return {
            "explanation_score": score.score,
            "has_command_breakdown": float(score.has_command_breakdown),
            "has_field_descriptions": float(score.has_field_descriptions),
            "has_usage_context": float(score.has_usage_context),
            "has_examples": float(score.has_examples),
            "technical_depth": score.technical_depth,
        }

    def _compute_text_metrics(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute ROUGE and BERTScore for text quality."""
        scores: dict[str, float] = {}

        # ROUGE scores
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

            for pred, ref in zip(predictions, references):
                result = scorer.score(ref, pred)
                for key in rouge_scores:
                    rouge_scores[key].append(result[key].fmeasure)

            for key, values in rouge_scores.items():
                scores[key] = sum(values) / len(values) if values else 0.0

        except ImportError:
            scores["rouge1"] = -1.0
            scores["rouge2"] = -1.0
            scores["rougeL"] = -1.0

        # BERTScore
        try:
            from bert_score import score as bert_score

            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            scores["bertscore_precision"] = P.mean().item()
            scores["bertscore_recall"] = R.mean().item()
            scores["bertscore_f1"] = F1.mean().item()

        except ImportError:
            scores["bertscore_f1"] = -1.0

        return scores

    def run(
        self,
        test_path: str | Path,
        system_prompt: str | None = None,
        progress: bool = True,
    ) -> EvalSummary:
        """Run full evaluation on a test set.

        Args:
            test_path: Path to JSONL test set.
            system_prompt: Optional system prompt for generation.
            progress: Show progress bar.

        Returns:
            EvalSummary with aggregated scores.
        """
        start_time = time.time()
        examples = self.load_test_set(test_path)
        self.results = []

        predictions_by_cat: dict[str, list[str]] = {}
        references_by_cat: dict[str, list[str]] = {}

        iterator = tqdm(examples, desc="Evaluating") if progress else examples

        for i, example in enumerate(iterator):
            instruction = example["instruction"]
            reference = example["response"]
            category = example.get("category", "unknown")

            # Generate prediction
            prediction = self.generate(instruction, system_prompt=system_prompt)

            # Category-specific scoring
            if category == "config_generation":
                scores = self._eval_config_example(prediction, reference)
            elif category == "troubleshooting":
                scores = self._eval_troubleshooting_example(prediction, reference)
            elif category == "command_explanation":
                scores = self._eval_explanation_example(prediction, reference)
            else:
                scores = {}

            result = EvalResult(
                example_id=f"eval_{i:04d}",
                category=category,
                instruction=instruction,
                reference=reference,
                prediction=prediction,
                scores=scores,
            )
            self.results.append(result)

            # Collect for text metrics
            predictions_by_cat.setdefault(category, []).append(prediction)
            references_by_cat.setdefault(category, []).append(reference)

        # Compute text metrics per category
        category_scores: dict[str, dict[str, float]] = {}
        all_predictions: list[str] = []
        all_references: list[str] = []

        for cat in predictions_by_cat:
            preds = predictions_by_cat[cat]
            refs = references_by_cat[cat]
            all_predictions.extend(preds)
            all_references.extend(refs)

            text_scores = self._compute_text_metrics(preds, refs)

            # Average domain-specific scores for this category
            cat_results = [r for r in self.results if r.category == cat]
            if cat_results:
                domain_scores: dict[str, list[float]] = {}
                for r in cat_results:
                    for key, val in r.scores.items():
                        domain_scores.setdefault(key, []).append(val)
                avg_domain = {k: sum(v) / len(v) for k, v in domain_scores.items()}
            else:
                avg_domain = {}

            category_scores[cat] = {**avg_domain, **text_scores}

        # Overall scores
        overall_text = self._compute_text_metrics(all_predictions, all_references)
        all_domain_scores: dict[str, list[float]] = {}
        for r in self.results:
            for key, val in r.scores.items():
                all_domain_scores.setdefault(key, []).append(val)
        overall_domain = {k: sum(v) / len(v) for k, v in all_domain_scores.items()}

        eval_time = time.time() - start_time

        return EvalSummary(
            total_examples=len(self.results),
            category_scores=category_scores,
            overall_scores={**overall_domain, **overall_text},
            eval_time_seconds=round(eval_time, 2),
            model_name=self.model_path,
        )

    def save_results(self, path: str | Path) -> Path:
        """Save detailed results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "results": [
                {
                    "example_id": r.example_id,
                    "category": r.category,
                    "instruction": r.instruction,
                    "reference": r.reference[:200] + "..." if len(r.reference) > 200 else r.reference,
                    "prediction": r.prediction[:200] + "..." if len(r.prediction) > 200 else r.prediction,
                    "scores": r.scores,
                }
                for r in self.results
            ],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        return path

    @staticmethod
    def generate_test_set(output_path: str | Path, size: int = 500) -> Path:
        """Generate a synthetic test set for evaluation.

        Creates a balanced test set with examples across all categories.
        """
        from netops_7b.data.curate import DatasetCurator
        from netops_7b.data.quality import QualityFilter

        curator = DatasetCurator(
            target_size=size,
            seed=12345,  # Different seed from training
        )
        pairs = curator.generate(progress=False)
        qf = QualityFilter()
        pairs = qf.filter(pairs, progress=False)

        output_path = Path(output_path)
        return curator.save(pairs[:size], output_path, fmt="jsonl")
