"""GGUF export for NetOps-7B model.

Exports the fine-tuned model to GGUF format for use with llama.cpp,
Ollama, LM Studio, and other local inference engines.

Supported quantization levels:
  - Q4_K_M: 4-bit (medium), best size/quality balance
  - Q5_K_M: 5-bit (medium), higher quality
  - Q8_0: 8-bit, near-lossless
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_QUANTS = ["q4_k_m", "q5_k_m", "q8_0"]


@dataclass
class GGUFExporter:
    """Export fine-tuned model to GGUF format.

    Uses either Unsloth's built-in export or llama.cpp's convert script.
    """

    model_path: str = "outputs/netops-7b-qlora/final"
    output_dir: str = "outputs/gguf"
    model_name: str = "netops-7b"

    def export_unsloth(
        self,
        quantization_methods: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export using Unsloth's GGUF export (preferred method).

        Args:
            quantization_methods: List of quantization methods.
                Defaults to ['q4_k_m', 'q5_k_m', 'q8_0'].

        Returns:
            Dict mapping quantization method to output file path.
        """
        from unsloth import FastLanguageModel

        quantization_methods = quantization_methods or SUPPORTED_QUANTS
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, Path] = {}

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )

        for quant in quantization_methods:
            quant_lower = quant.lower()
            if quant_lower not in SUPPORTED_QUANTS:
                raise ValueError(
                    f"Unsupported quantization: {quant}. "
                    f"Supported: {SUPPORTED_QUANTS}"
                )

            output_file = output_dir / f"{self.model_name}-{quant_lower}.gguf"

            model.save_pretrained_gguf(
                str(output_dir),
                tokenizer,
                quantization_method=quant_lower,
            )

            exported[quant_lower] = output_file

        return exported

    def export_llamacpp(
        self,
        llama_cpp_path: str = "llama.cpp",
        quantization_methods: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export using llama.cpp convert scripts (fallback method).

        Requires llama.cpp to be built locally.

        Args:
            llama_cpp_path: Path to llama.cpp directory.
            quantization_methods: List of quantization methods.

        Returns:
            Dict mapping quantization method to output file path.
        """
        quantization_methods = quantization_methods or SUPPORTED_QUANTS
        llama_path = Path(llama_cpp_path)
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, Path] = {}

        # Step 1: Convert to f16 GGUF
        f16_path = output_dir / f"{self.model_name}-f16.gguf"
        convert_script = llama_path / "convert_hf_to_gguf.py"

        if not convert_script.exists():
            raise FileNotFoundError(
                f"llama.cpp convert script not found at {convert_script}. "
                f"Clone llama.cpp and build it first."
            )

        subprocess.run(
            [
                "python", str(convert_script),
                self.model_path,
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Step 2: Quantize to each level
        quantize_bin = llama_path / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = llama_path / "llama-quantize"

        for quant in quantization_methods:
            quant_upper = quant.upper()
            output_file = output_dir / f"{self.model_name}-{quant.lower()}.gguf"

            subprocess.run(
                [
                    str(quantize_bin),
                    str(f16_path),
                    str(output_file),
                    quant_upper,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            exported[quant.lower()] = output_file

        return exported

    def create_ollama_modelfile(
        self,
        gguf_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate an Ollama Modelfile for the exported GGUF.

        Args:
            gguf_path: Path to the GGUF file.
            output_path: Where to write the Modelfile. Defaults to same dir as GGUF.

        Returns:
            Path to the Modelfile.
        """
        gguf_path = Path(gguf_path)
        output_path = Path(output_path) if output_path else gguf_path.parent / "Modelfile"

        modelfile_content = f'''FROM {gguf_path.name}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """You are NetOps-7B, an expert network operations assistant specialized in Cisco, Juniper, and Arista network equipment. You help network engineers with:

1. **Configuration generation** — Creating vendor-specific device configurations
2. **Troubleshooting** — Diagnosing network issues systematically
3. **Command explanation** — Explaining CLI commands and their output

Provide accurate, detailed, and actionable responses. When generating configurations, use proper syntax for the target platform. When troubleshooting, follow a systematic approach from Layer 1 upward."""

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>"""
'''

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(modelfile_content)

        return output_path

    def get_file_sizes(self) -> dict[str, str]:
        """Get human-readable file sizes for all exported GGUFs."""
        output_dir = Path(self.output_dir)
        sizes: dict[str, str] = {}

        for gguf_file in output_dir.glob("*.gguf"):
            size_bytes = gguf_file.stat().st_size
            if size_bytes >= 1_073_741_824:
                sizes[gguf_file.name] = f"{size_bytes / 1_073_741_824:.1f} GB"
            else:
                sizes[gguf_file.name] = f"{size_bytes / 1_048_576:.1f} MB"

        return sizes
