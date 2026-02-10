<div align="center">

# From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-orange)](https://huggingface.co/FreedomIntelligence/TierMem)

**A memory-augmented LLM system for long-context question answering with intelligent routing between summary-based and raw-retrieval pipelines.**

[Installation](#installation) • [Quick Start](#quick-start) • [Model](#pretrained-model) • [Benchmarks](#supported-benchmarks) • [Training](#router-training)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Pretrained Model](#pretrained-model)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Router Training](#router-training)
- [Supported Benchmarks](#supported-benchmarks)
- [Results](#results)
- [Configuration](#configuration)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

TierMem implements a two-tier memory architecture that balances efficiency and accuracy for long-context question answering:

1. **Summary Index (S-path)**: Fast semantic search over extracted facts using Mem0
2. **Page Store (R-path)**: Raw conversation chunks with BM25 retrieval for detailed context

A trained router model dynamically selects between these paths based on query complexity, ensuring optimal performance across different types of questions.

## Key Features

✨ **Intelligent Routing** - Trained router automatically selects the best retrieval strategy
🚀 **High Performance** - Optimized for long conversations with 100K+ tokens
🎯 **Dual Retrieval Paths** - Combines semantic search and keyword-based retrieval
🔄 **Provenance Tracking** - Maintains memory source verification and lineage
📊 **Multi-Benchmark Support** - Evaluated on LoCoMo, LongMemEval, MemoryAgentBench, and more
⚡ **Concurrent Processing** - Multi-worker support for batch evaluation
🧠 **LLM-Agnostic** - Works with any OpenAI-compatible API

## Architecture

<div align="center">
  <img src="frame.jpg" alt="TierMem Architecture" width="800"/>
  <p><em>Two-tier memory system with intelligent routing between summary and raw retrieval paths</em></p>
</div>

---


## Installation

### Prerequisites

- Python 3.10+
- Qdrant vector database
- CUDA-capable GPU (for router training)

### Setup

```bash
# Clone the repository
git clone https://github.com/FreedomIntelligence/TierMem.git
cd TierMem

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
./start_qdrant.sh

# Set environment variables
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=your_base_url  # Optional
```

## Model

Our trained router model is available on HuggingFace:

<div align="center">

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-FreedomIntelligence%2FTierMem-orange?style=for-the-badge)](https://huggingface.co/FreedomIntelligence/TierMem)

**Download:** `https://huggingface.co/FreedomIntelligence/TierMem`

</div>

The router model is a fine-tuned classifier that determines whether to use the Summary (S-path) or Raw (R-path) retrieval pipeline based on query characteristics.

---

## Quick Start

### Running Benchmarks

**LoCoMo Benchmark** (Concurrent Execution)
```bash
python test_TierMem_locomo_multi.py --limit 10 --max-workers 4
```

**LongMemEval Benchmark**
```bash
python test_TierMem_longmemeval_multi.py --limit 10 --max-workers 4
```

**MemoryAgentBench**
```bash
python test_TierMem_memoryagentbench.py \
  --split Accurate_Retrieval \
  --limit 10 \
  --max-workers 4
```

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--limit N` | Process only N sessions | All sessions |
| `--max-workers N` | Number of concurrent workers | 4 |
| `--model MODEL` | LLM model name | `gpt-4o-mini` |
| `--run-id ID` | Custom run identifier | auto-generated |

> **Tip:** Start with a small `--limit` value to test your setup before running full benchmarks.

## Project Structure

```
TierMem/
├── core/                           # Benchmark framework
│   ├── systems/                   # Memory system interfaces
│   ├── datasets/                  # Dataset loaders (LoCoMo, LongMemEval, etc.)
│   └── runner/                    # Evaluation runners
├── src/                           # System implementations
│   ├── memory/                   # Memory system implementations
│   ├── linked_view/              # TierMem architecture components
│   ├── evaluation/               # LLM-as-Judge evaluation
│   └── mem0/                     # Modified mem0 library
├── scripts/
│   └── router_training/          # Router model training pipeline
├── test_TierMem_*.py             # Benchmark runner scripts
└── start_qdrant.sh               # Qdrant startup script
```

## Router Training

The router model determines which retrieval path (Summary or Raw) to use for each query. Below is the complete training pipeline:

### Training Pipeline

**Step 1: Build Offline Dataset**
```bash
python scripts/router_training/1_build_offline_dataset.py
```
Generates training data by running both retrieval paths on sample queries.

**Step 2: Prepare SFT Data**
```bash
python scripts/router_training/2_prepare_sft_data_v2.py
```
Formats the offline dataset for supervised fine-tuning.

**Step 3: Supervised Fine-Tuning (SFT)**
```bash
sbatch scripts/router_training/train_router_sft.sbatch
```
Fine-tunes the base model to classify queries.

**Step 4: GRPO Training** *(Optional)*
```bash
sbatch scripts/router_training/train_router_grpo.sbatch
```
Further optimizes the router using reinforcement learning.

**Step 5: Deploy Router with vLLM**
```bash
sbatch scripts/router_training/start_router_vllm.sbatch
```
Serves the trained router for inference.

### Configuration

Before running the training scripts, update the following placeholders in the `.sbatch` files:
- `<PROJECT_ROOT>`: Path to your TierMem installation
- `<MS_SWIFT_DIR>`: Path to ms-swift installation

> **Note:** Training dependencies (ms-swift, deepspeed, etc.) are listed in `requirements.txt` as optional. Uncomment them if you plan to train the router.

---

## Supported Benchmarks

TierMem has been evaluated on multiple long-context memory benchmarks:

| Benchmark | Description | Metrics | Script |
|-----------|-------------|---------|--------|
| **LoCoMo** | Long-context memory QA | F1, Accuracy | `test_TierMem_locomo_multi.py` |
| **LongMemEval** | Long memory evaluation | F1, Accuracy | `test_TierMem_longmemeval_multi.py` |
| **MemoryAgentBench** | Multi-split agent benchmark | F1, Accuracy | `test_TierMem_memoryagentbench.py` |
| **HotPotQA** | Multi-hop reasoning QA | F1, EM | *(Coming soon)* |
| **HaluMem** | Hallucination evaluation | Accuracy | *(Coming soon)* |

---

## Results

Results are saved to `results/{benchmark}/{system_name}/{run_id}/`:

```
results/locomo/linked_view/my_run/
├── sessions/
│   ├── conv-1_write.jsonl
│   ├── conv-1_qa.jsonl
│   └── ...
├── summary.json              # Aggregated metrics
└── eval_details.json         # Detailed evaluation
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `OPENAI_BASE_URL` | Custom API endpoint | No |
| `QDRANT_HOST` | Qdrant server host | No (default: localhost) |
| `QDRANT_PORT` | Qdrant server port | No (default: 6333) |

## Citation

If you use TierMem in your research, please cite our work:

```bibtex
@article{tiermem2026,
  title={From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents},
  author={Your Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

> **Note:** Please update this citation once the paper is published.

---

## Contributing

We welcome contributions! Here's how you can help:

- **Report Bugs**: Open an issue describing the bug and how to reproduce it
- **Suggest Features**: Share your ideas for new features or improvements
- **Submit PRs**: Fix bugs, add features, or improve documentation
- **Improve Docs**: Help us make the documentation clearer and more comprehensive

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software for both commercial and non-commercial purposes.

## Acknowledgments

This project builds on excellent open-source work:

- [**Mem0**](https://github.com/mem0ai/mem0) - Memory management and semantic extraction
- [**ms-swift**](https://github.com/modelscope/swift) - Efficient model training framework
- [**vLLM**](https://github.com/vllm-project/vllm) - High-performance LLM inference
- [**Qdrant**](https://github.com/qdrant/qdrant) - Vector similarity search engine

---

<div align="center">

**⭐ If you find TierMem useful, please consider giving us a star! ⭐**

Made with ❤️ by the FreedomIntelligence Team

</div>
