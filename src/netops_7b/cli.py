"""CLI interface for NetOps-7B pipeline.

Commands:
    netops data curate     — Generate and curate training dataset
    netops data convert    — Convert between Alpaca/ShareGPT formats
    netops train run       — Run QLoRA fine-tuning
    netops train export    — Export to GGUF format
    netops eval run        — Run evaluation harness
    netops eval benchmark  — Run base vs fine-tuned comparison
    netops eval test-set   — Generate evaluation test set
    netops demo launch     — Launch Gradio demo
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="netops-7b")
def cli():
    """NetOps-7B: QLoRA fine-tuning pipeline for network operations."""
    pass


# ---------------------------------------------------------------------------
# Data commands
# ---------------------------------------------------------------------------

@cli.group()
def data():
    """Dataset curation and processing commands."""
    pass


@data.command()
@click.option("--output", "-o", default="data/train.jsonl", help="Output file path")
@click.option("--size", "-n", default=20000, help="Target dataset size")
@click.option("--format", "fmt", type=click.Choice(["jsonl", "json"]), default="jsonl")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def curate(output: str, size: int, fmt: str, seed: int):
    """Generate and curate a synthetic training dataset."""
    from netops_7b.data.curate import DatasetCurator

    console.print(f"[bold blue]Curating dataset[/] — target size: {size}")
    curator = DatasetCurator(target_size=size, seed=seed)
    path = curator.curate(output, fmt=fmt)

    # Show stats
    pairs = curator.generate(progress=False)
    stats = curator.stats(pairs)

    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total examples", str(stats["total"]))
    table.add_row("Avg length (chars)", f"{stats['avg_length_chars']:.0f}")

    for cat, count in stats["categories"].items():
        table.add_row(f"  {cat}", str(count))

    console.print(table)
    console.print(f"\n[bold green]Dataset saved to {path}[/]")


@data.command()
@click.option("--input", "-i", "input_path", required=True, help="Input JSONL file")
@click.option("--output", "-o", "output_path", required=True, help="Output file path")
@click.option("--format", "fmt", type=click.Choice(["alpaca", "sharegpt"]), default="sharegpt")
@click.option("--system-prompt", default=None, help="Custom system prompt")
def convert(input_path: str, output_path: str, fmt: str, system_prompt: str | None):
    """Convert dataset between Alpaca and ShareGPT formats."""
    from netops_7b.data.formats import convert_file

    console.print(f"[bold blue]Converting[/] {input_path} → {fmt} format")
    path = convert_file(input_path, output_path, output_format=fmt, system_prompt=system_prompt)
    console.print(f"[bold green]Saved to {path}[/]")


# ---------------------------------------------------------------------------
# Training commands
# ---------------------------------------------------------------------------

@cli.group()
def train():
    """Training and export commands."""
    pass


@train.command("run")
@click.option("--config", "-c", default=None, help="Path to training config JSON")
@click.option("--dataset", "-d", default="data/train.jsonl", help="Training dataset path")
@click.option("--output", "-o", default="outputs/netops-7b-qlora", help="Output directory")
@click.option("--epochs", default=3, help="Number of training epochs")
@click.option("--lr", default=2e-4, help="Learning rate")
@click.option("--batch-size", default=2, help="Per-device batch size")
@click.option("--lora-rank", default=16, help="LoRA rank")
def train_run(config: str | None, dataset: str, output: str, epochs: int,
              lr: float, batch_size: int, lora_rank: int):
    """Run QLoRA fine-tuning."""
    from netops_7b.train.qlora import QLoRAConfig, QLoRATrainer

    if config:
        cfg = QLoRAConfig.load(config)
    else:
        cfg = QLoRAConfig(
            dataset_path=dataset,
            output_dir=output,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            lora_rank=lora_rank,
        )

    console.print("[bold blue]Starting QLoRA fine-tuning[/]")
    console.print(f"  Base model: {cfg.base_model}")
    console.print(f"  LoRA rank: {cfg.lora_rank}")
    console.print(f"  Epochs: {cfg.num_train_epochs}")
    console.print(f"  Effective batch size: {cfg.effective_batch_size}")
    console.print(f"  Learning rate: {cfg.learning_rate}")

    trainer = QLoRATrainer(cfg)
    trainer.setup()

    params = trainer.print_trainable_params()
    console.print(f"  Trainable params: {params['trainable_params']:,} ({params['trainable_percent']}%)")

    metrics = trainer.train()
    save_path = trainer.save()

    console.print(f"\n[bold green]Training complete![/]")
    console.print(f"  Loss: {metrics.get('train_loss', 'N/A')}")
    console.print(f"  Saved to: {save_path}")


@train.command()
@click.option("--model", "-m", default="outputs/netops-7b-qlora/final", help="Model path")
@click.option("--output", "-o", default="outputs/gguf", help="Output directory")
@click.option("--quant", "-q", multiple=True, default=["q4_k_m", "q5_k_m", "q8_0"],
              help="Quantization methods")
@click.option("--method", type=click.Choice(["unsloth", "llamacpp"]), default="unsloth")
def export(model: str, output: str, quant: tuple[str, ...], method: str):
    """Export model to GGUF format."""
    from netops_7b.train.export import GGUFExporter

    exporter = GGUFExporter(model_path=model, output_dir=output)

    console.print(f"[bold blue]Exporting to GGUF[/] — method: {method}")
    console.print(f"  Quantizations: {', '.join(quant)}")

    if method == "unsloth":
        exported = exporter.export_unsloth(quantization_methods=list(quant))
    else:
        exported = exporter.export_llamacpp(quantization_methods=list(quant))

    for q, path in exported.items():
        console.print(f"  [green]{q}[/]: {path}")

    console.print("[bold green]Export complete![/]")


# ---------------------------------------------------------------------------
# Evaluation commands
# ---------------------------------------------------------------------------

@cli.group("eval")
def eval_group():
    """Evaluation and benchmarking commands."""
    pass


@eval_group.command("run")
@click.option("--model", "-m", default="outputs/netops-7b-qlora/final", help="Model path")
@click.option("--test-set", "-t", default="data/test.jsonl", help="Test set path")
@click.option("--output", "-o", default="outputs/eval_results.json", help="Results output path")
def eval_run(model: str, test_set: str, output: str):
    """Run evaluation harness on the fine-tuned model."""
    from netops_7b.eval.harness import EvaluationHarness

    console.print(f"[bold blue]Running evaluation[/] — model: {model}")

    harness = EvaluationHarness(model_path=model)
    harness.load_model()
    summary = harness.run(test_set)
    harness.save_results(output)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for metric, score in sorted(summary.overall_scores.items()):
        table.add_row(metric, f"{score:.4f}")

    console.print(table)
    console.print(f"\n[bold green]Results saved to {output}[/]")


@eval_group.command("benchmark")
@click.option("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
@click.option("--finetuned", default="outputs/netops-7b-qlora/final", help="Fine-tuned model")
@click.option("--test-set", "-t", default="data/test.jsonl", help="Test set path")
@click.option("--output", "-o", default="outputs/benchmark.json", help="Results output path")
def benchmark(base_model: str, finetuned: str, test_set: str, output: str):
    """Run comparison benchmark: base model vs fine-tuned."""
    from netops_7b.eval.benchmark import BenchmarkRunner

    console.print("[bold blue]Running benchmark comparison[/]")

    runner = BenchmarkRunner(base_model=base_model, finetuned_model=finetuned)
    result = runner.run(test_set)
    runner.save(result, output)

    console.print("\n" + result.to_markdown_table())
    console.print(f"\n[bold green]Benchmark saved to {output}[/]")


@eval_group.command("test-set")
@click.option("--output", "-o", default="data/test.jsonl", help="Output path")
@click.option("--size", "-n", default=500, help="Number of test examples")
def gen_test_set(output: str, size: int):
    """Generate a synthetic evaluation test set."""
    from netops_7b.eval.harness import EvaluationHarness

    console.print(f"[bold blue]Generating test set[/] — size: {size}")
    path = EvaluationHarness.generate_test_set(output, size=size)
    console.print(f"[bold green]Test set saved to {path}[/]")


# ---------------------------------------------------------------------------
# Demo command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model", "-m", default="cwccie/netops-7b", help="Model path or HF repo")
@click.option("--port", "-p", default=7860, help="Server port")
@click.option("--share", is_flag=True, help="Create public Gradio link")
def demo(model: str, port: int, share: bool):
    """Launch the Gradio demo interface."""
    from netops_7b.demo.gradio_app import NetOpsDemo

    console.print(f"[bold blue]Launching NetOps-7B demo[/] — model: {model}")
    app = NetOpsDemo(model_path=model)
    app.load_model()
    app.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    cli()
