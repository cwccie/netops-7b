# NetOps-7B

**A QLoRA fine-tuned language model for network operations — configuration generation, troubleshooting, and command explanation for Cisco, Juniper, and Arista platforms.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Card-blue)](https://huggingface.co/cwccie/netops-7b)
[![CI](https://github.com/cwccie/netops-7b/actions/workflows/ci.yml/badge.svg)](https://github.com/cwccie/netops-7b/actions/workflows/ci.yml)

> **Companion repository** for the [NetOps-7B model on Hugging Face](https://huggingface.co/cwccie/netops-7b).
> Contains the complete training pipeline, evaluation harness, and demo interface.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        NetOps-7B Pipeline                        │
├──────────────┬───────────────┬────────────────┬─────────────────┤
│  Data        │  Training     │  Evaluation    │  Deployment     │
│              │               │                │                 │
│  Template    │  Qwen 2.5 7B  │  Config syntax │  Gradio demo    │
│  synthesis   │  Instruct     │  validation    │                 │
│  (15-30K)    │       │       │                │  GGUF export    │
│       │      │  QLoRA 4-bit  │  Troubleshoot  │  (Q4/Q5/Q8)    │
│  Quality     │  LoRA r=16    │  accuracy      │                 │
│  filtering   │  all linear   │                │  Ollama         │
│       │      │  3 epochs     │  ROUGE /       │  Modelfile      │
│  Alpaca /    │  lr=2e-4      │  BERTScore     │                 │
│  ShareGPT    │  Unsloth      │                │  HF Hub push    │
└──────────────┴───────────────┴────────────────┴─────────────────┘
```

## Quickstart

### Installation

```bash
pip install git+https://github.com/cwccie/netops-7b.git
```

Or clone and install in development mode:

```bash
git clone https://github.com/cwccie/netops-7b.git
cd netops-7b
pip install -e ".[dev]"
```

### Full Pipeline

```bash
# 1. Generate and curate training data (CPU only)
netops data curate -o data/train.jsonl -n 20000

# 2. Convert to ShareGPT format
netops data convert -i data/train.jsonl -o data/train_sharegpt.json --format sharegpt

# 3. Run QLoRA fine-tuning (requires GPU)
netops train run -d data/train.jsonl -o outputs/qlora --epochs 3 --lr 2e-4

# 4. Export to GGUF
netops train export -m outputs/qlora/final -q q4_k_m -q q5_k_m -q q8_0

# 5. Generate test set and evaluate
netops eval test-set -o data/test.jsonl -n 500
netops eval run -m outputs/qlora/final -t data/test.jsonl

# 6. Run benchmark comparison vs base model
netops eval benchmark --base-model Qwen/Qwen2.5-7B-Instruct --finetuned outputs/qlora/final

# 7. Launch interactive demo
netops demo -m outputs/qlora/final --port 7860
```

### Docker

```bash
# Build
docker compose build

# Run the full pipeline
docker compose run curate       # Generate data
docker compose run train        # Fine-tune (GPU required)
docker compose run eval         # Evaluate (GPU required)
docker compose up demo          # Launch demo at localhost:7860

# Run tests
docker compose run test
```

---

## Model Card

### Overview

| Property | Value |
|----------|-------|
| **Model Name** | NetOps-7B |
| **Base Model** | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| **Fine-tuning Method** | QLoRA (4-bit NF4) via [Unsloth](https://github.com/unslothai/unsloth) |
| **Parameters** | ~7B total, ~26M trainable (LoRA) |
| **Context Length** | 4096 tokens |
| **License** | MIT |
| **Author** | Corey A. Wade (CCIE #14124, CISSP, PhD) |

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | 4-bit NormalFloat (NF4) with double quantization |
| Epochs | 3 |
| Learning Rate | 2e-4 (cosine schedule) |
| Batch Size | 2 per device, 4 gradient accumulation (effective 8) |
| Optimizer | AdamW 8-bit |
| Warmup | 5% of training steps |
| Sequence Packing | Enabled |

### Intended Use

NetOps-7B is designed for network engineers and operators who need:

- **Configuration generation** — Create vendor-specific device configurations from natural language descriptions
- **Troubleshooting assistance** — Systematic diagnosis of network issues with verification commands
- **Command reference** — Detailed explanations of CLI commands with field descriptions and examples

**Supported platforms:** Cisco IOS/IOS-XE/IOS-XR/NX-OS, Juniper Junos, Arista EOS

### Limitations

- **Not production-ready without human review** — Always validate generated configurations before deployment
- **Training data is synthetic** — Generated from templates, not real-world documentation
- **No real-time network awareness** — Cannot query live devices or detect current state
- **Vendor coverage** — Focused on Cisco, Juniper, and Arista; limited coverage of other vendors (Palo Alto, Fortinet, etc.)
- **Knowledge cutoff** — Based on Qwen 2.5 7B's training data; may not reflect latest software versions

### Ethical Considerations

- Generated configurations should always be reviewed by qualified network engineers
- The model may generate configurations with security implications (e.g., weak ACLs)
- Not intended for autonomous network management without human oversight

---

## Dataset

### Composition

The training dataset is synthetically generated with the following distribution:

| Category | Proportion | Description |
|----------|-----------|-------------|
| Cisco Configuration | 25% | IOS, NX-OS, IOS-XR, IOS-XE configs (OSPF, BGP, VXLAN, ACLs, HSRP, IS-IS, SD-WAN) |
| Juniper Configuration | 15% | Junos set-style configs (OSPF, BGP, EVPN, firewalls, CoS, routing-instances) |
| Arista Configuration | 15% | EOS configs (BGP, VXLAN, MLAG, prefix-lists, route-maps) |
| Troubleshooting | 25% | Multi-vendor diagnostic workflows (interface flapping, BGP issues, OSPF, EVPN, packet loss) |
| Command Explanation | 20% | CLI command reference with field descriptions, usage context, and examples |

### Quality Pipeline

1. **Template synthesis** — Parameterized templates with realistic network values
2. **Deduplication** — SHA-256 hash-based exact dedup
3. **Length filtering** — Instruction: 20-2000 chars, Response: 50-8000 chars
4. **Domain validation** — Must contain network-domain keywords
5. **Toxicity screening** — Rejects malicious instruction patterns
6. **PII detection** — Filters entries with email addresses, phone numbers, etc.

---

## Evaluation

### Methodology

The evaluation harness measures three dimensions:

1. **Configuration Syntax Validity** — Regex-based structural parsing against vendor-specific patterns (interface declarations, routing protocol blocks, ACLs, etc.)

2. **Troubleshooting Quality** — Composite score based on:
   - Systematic approach (step-by-step methodology)
   - Verification commands (show/debug commands included)
   - Root cause analysis (identifies likely causes)
   - Remediation steps (actionable fixes)
   - OSI layer coverage (L1 through L7)

3. **Explanation Quality** — Composite of:
   - ROUGE-1/2/L scores against reference explanations
   - BERTScore F1 for semantic similarity
   - Domain metrics: command breakdown, field descriptions, usage context

### Results (Placeholder)

> Results will be populated after training is complete.

| Metric | Base (Qwen 2.5 7B) | NetOps-7B | Delta |
|--------|-------------------|-----------|-------|
| Config Syntax Score | — | — | — |
| Troubleshooting Score | — | — | — |
| Explanation Score | — | — | — |
| ROUGE-L | — | — | — |
| BERTScore F1 | — | — | — |

---

## GGUF Downloads

After training, the model is exported to GGUF format for local inference:

| Quantization | Use Case | Est. Size |
|-------------|----------|-----------|
| Q4_K_M | Best balance of size and quality | ~4.4 GB |
| Q5_K_M | Higher quality, moderate size | ~5.1 GB |
| Q8_0 | Near-lossless, largest | ~7.7 GB |

### Using with Ollama

```bash
# Create from GGUF
ollama create netops-7b -f Modelfile

# Run
ollama run netops-7b "Configure OSPF area 0 on a Cisco router with interface Gi0/0"
```

### Using with llama.cpp

```bash
./llama-cli -m netops-7b-q4_k_m.gguf \
  -p "Configure a Junos BGP session with neighbor 10.0.0.1 in AS 65001" \
  -n 512 --temp 0.3
```

---

## Project Structure

```
netops-7b/
├── src/netops_7b/
│   ├── data/
│   │   ├── curate.py        # Synthetic data generation (templates + parameter filling)
│   │   ├── formats.py       # Alpaca / ShareGPT format converters
│   │   └── quality.py       # Quality filtering (dedup, length, toxicity, PII)
│   ├── train/
│   │   ├── qlora.py         # QLoRA config + trainer (Unsloth + TRL)
│   │   └── export.py        # GGUF export (Q4_K_M, Q5_K_M, Q8_0)
│   ├── eval/
│   │   ├── metrics.py       # Custom network domain metrics
│   │   ├── harness.py       # Evaluation harness (syntax, troubleshooting, ROUGE/BERT)
│   │   └── benchmark.py     # Base vs fine-tuned comparison
│   ├── demo/
│   │   └── gradio_app.py    # Gradio web interface
│   └── cli.py               # Click CLI for all pipeline stages
├── tests/                    # Comprehensive test suite
├── notebooks/
│   ├── 01_data_curation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── pyproject.toml            # Package config (hatchling)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml  # Lint (ruff) + test (pytest)
```

---

## Notebooks

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| [01_data_curation.ipynb](notebooks/01_data_curation.ipynb) | Generate and curate training data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwccie/netops-7b/blob/main/notebooks/01_data_curation.ipynb) |
| [02_training.ipynb](notebooks/02_training.ipynb) | QLoRA fine-tuning with Unsloth | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwccie/netops-7b/blob/main/notebooks/02_training.ipynb) |
| [03_evaluation.ipynb](notebooks/03_evaluation.ipynb) | Evaluation and benchmarking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwccie/netops-7b/blob/main/notebooks/03_evaluation.ipynb) |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

---

## Author

**Corey A. Wade**
- CCIE #14124 | CISSP | PhD (AI + Security)
- GitHub: [@cwccie](https://github.com/cwccie)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@misc{wade2026netops7b,
  title={NetOps-7B: A QLoRA Fine-Tuned Model for Network Operations},
  author={Wade, Corey A.},
  year={2026},
  url={https://github.com/cwccie/netops-7b},
  note={Companion repository for HuggingFace model cwccie/netops-7b}
}
```
