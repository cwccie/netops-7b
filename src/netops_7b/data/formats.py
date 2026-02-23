"""Format converters for instruction-tuning datasets.

Supports Alpaca and ShareGPT formats, the two most common formats
for supervised fine-tuning with libraries like TRL and Axolotl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netops_7b.data.curate import InstructionPair


ALPACA_SYSTEM_PROMPT = (
    "You are NetOps-7B, an expert network operations assistant trained on Cisco, "
    "Juniper, and Arista documentation. You help network engineers with configuration "
    "generation, troubleshooting, and command explanation. Provide accurate, detailed, "
    "and vendor-specific responses."
)


@dataclass
class AlpacaFormatter:
    """Convert instruction pairs to Alpaca format.

    Alpaca format:
    {
        "instruction": "...",
        "input": "",
        "output": "...",
        "system": "..."  (optional)
    }
    """

    system_prompt: str = ALPACA_SYSTEM_PROMPT
    include_system: bool = True

    def format_one(self, pair: InstructionPair) -> dict[str, str]:
        """Convert a single instruction pair to Alpaca format."""
        result: dict[str, str] = {
            "instruction": pair.instruction,
            "input": "",
            "output": pair.response,
        }
        if self.include_system:
            result["system"] = self.system_prompt
        return result

    def format_batch(self, pairs: list[InstructionPair]) -> list[dict[str, str]]:
        """Convert a batch of instruction pairs to Alpaca format."""
        return [self.format_one(p) for p in pairs]

    def save(self, pairs: list[InstructionPair], output_path: str | Path) -> Path:
        """Format and save to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatted = self.format_batch(pairs)
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        return output_path

    def save_jsonl(self, pairs: list[InstructionPair], output_path: str | Path) -> Path:
        """Format and save to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(self.format_one(pair)) + "\n")
        return output_path


@dataclass
class ShareGPTFormatter:
    """Convert instruction pairs to ShareGPT multi-turn format.

    ShareGPT format:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """

    system_prompt: str = ALPACA_SYSTEM_PROMPT
    include_system: bool = True

    def format_one(self, pair: InstructionPair) -> dict[str, Any]:
        """Convert a single instruction pair to ShareGPT format."""
        conversations: list[dict[str, str]] = []

        if self.include_system:
            conversations.append({
                "from": "system",
                "value": self.system_prompt,
            })

        conversations.extend([
            {"from": "human", "value": pair.instruction},
            {"from": "gpt", "value": pair.response},
        ])

        return {"conversations": conversations}

    def format_multi_turn(self, pairs: list[InstructionPair]) -> dict[str, Any]:
        """Combine multiple instruction pairs into a single multi-turn conversation.

        Useful for creating realistic multi-turn troubleshooting dialogues.
        """
        conversations: list[dict[str, str]] = []

        if self.include_system:
            conversations.append({
                "from": "system",
                "value": self.system_prompt,
            })

        for pair in pairs:
            conversations.extend([
                {"from": "human", "value": pair.instruction},
                {"from": "gpt", "value": pair.response},
            ])

        return {"conversations": conversations}

    def format_batch(self, pairs: list[InstructionPair]) -> list[dict[str, Any]]:
        """Convert a batch of instruction pairs to ShareGPT format (one conversation each)."""
        return [self.format_one(p) for p in pairs]

    def save(self, pairs: list[InstructionPair], output_path: str | Path) -> Path:
        """Format and save to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        formatted = self.format_batch(pairs)
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        return output_path

    def save_jsonl(self, pairs: list[InstructionPair], output_path: str | Path) -> Path:
        """Format and save to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(self.format_one(pair)) + "\n")
        return output_path


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    output_format: str = "sharegpt",
    system_prompt: str | None = None,
) -> Path:
    """Convert a JSONL file of instruction pairs to the specified format.

    Args:
        input_path: Path to input JSONL file (one InstructionPair dict per line).
        output_path: Path to write the formatted output.
        output_format: 'alpaca' or 'sharegpt'.
        system_prompt: Custom system prompt (uses default if None).

    Returns:
        Path to the output file.
    """
    input_path = Path(input_path)
    pairs: list[InstructionPair] = []

    with open(input_path) as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append(InstructionPair(
                instruction=data["instruction"],
                response=data["response"],
                category=data.get("category", "unknown"),
                vendor=data.get("vendor", "unknown"),
                complexity=data.get("complexity", "unknown"),
            ))

    kwargs: dict[str, Any] = {}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt

    if output_format == "alpaca":
        formatter = AlpacaFormatter(**kwargs)
    elif output_format == "sharegpt":
        formatter = ShareGPTFormatter(**kwargs)
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'alpaca' or 'sharegpt'.")

    return formatter.save(pairs, output_path)
