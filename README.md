# HDLxGraph

## Overview

HDLxGraph is a graph-based Retrieval-Augmented Generation (RAG) system specialized for Hardware Description Languages (HDL) like Verilog. It leverages knowledge graphs to improve code understanding, retrieval, and generation for hardware design tasks.

## Project Structure

- `rag/`: RAG implementation including Neo4j-based graph retrieval
- `generation/`: Code generation interfaces for various LLMs
- `benchmark/`: Evaluation benchmarks for HDL code search tasks
- `run.py`: Main script for running different tasks

## Getting Started

### Prerequisites

- Python 3.8+
- Neo4j Database (for HDLxGraph retrieval)
- API keys for supported LLMs (Claude, Qwen)
- Pyverilog for Verilog parsing

### Installation

1. Clone the repository
2. Install dependencies
3. Configure API keys in `generation/generation.py`

### Usage

Run tasks using the main script:

```bash
python run.py <task> <model> <RAG>
```

Parameters:

- `task`: `code_search`, `code_debugging`, or `code_completion`
- `model`: `llama`, `starcoder2-7b`, `claude`, or `qwen`
- `RAG`: `no-rag`, `bm25`, `similarity`, or `HDLxGraph`

Example:

```bash
python run.py code_search claude HDLxGraph
```

## Evaluation

The system is evaluated on multiple benchmarks:

- Code search performance using Mean Reciprocal Rank (MRR)
- Code debugging effectiveness
- Code completion quality using Verilog-eval
