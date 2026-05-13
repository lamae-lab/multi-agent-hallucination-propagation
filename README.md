# Multi-Agent Hallucination Propagation

Research framework for studying how hallucinations propagate through multi-agent LLM systems with different architectural patterns.

## Overview

This project investigates how false facts injected into one agent spread to downstream agents across five distinct architecture families. It introduces a **two-layer evaluation framework**:

- **Layer 1 (Established Metrics):** Faithfulness, Answer Correctness, Hallucination Rate, FActScore-style decomposition.
- **Layer 2 (Novel Propagation Metrics):** Survival Rate, Propagation Distance, Amplification Factor, Mutation Rate.

## Agent Architecture Families

| Family | Architecture | Description |
|--------|-------------|-------------|
| **F1** | ReAct | Single-agent baseline with Think-Act-Observe loop |
| **F2** | LangGraph State Graph | Sequential supervisor-managed agents with shared state |
| **F3** | Hierarchical Crew | Manager + specialists (researcher, analyst) with tool-call delegation |
| **F4** | Group Chat | Multi-agent consensus via AutoGen RoundRobinGroupChat |
| **F5** | Handoff Swarm | Autonomous agent handoffs via OpenAI Agents SDK |

## Injection Modes

- **Mode 1 (Synthetic):** Controlled false statements injected via modified system prompts.
- **Mode 2 (HaluEval):** Ecologically valid hallucinations from the HaluEval dataset.

## Datasets

- **HotpotQA** — Multi-hop question answering (distractor setting)
- **GSM8K** — Mathematical reasoning
- **HaluEval** — Validated hallucination examples
- **RAGCare-QA** — Cardiology MCQs for medical domain application

## Project Structure

```
src/
  agents/         BaseAgent with ReAct loop and tool support
  families/       F1-F5 architecture implementations
  injection/      Mode 1 & Mode 2 hallucination injection
  metrics/        Layer 1 (established) + Layer 2 (propagation) metrics
  tasks/          Dataset loaders (HotpotQA, GSM8K, HaluEval, RAGCare-QA)
  analysis/       Statistical tests (Wilcoxon, Cohen's d, bootstrap CI)
  llm/            LLM provider client and configuration
  tracing/        JSONL trace recording
  ui/             Streamlit dashboard
configs/          YAML configs for families and injection modes
experiments/      Study runners and results
```

## Setup

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Usage

### Run the main study

```bash
python experiments/run_main_study.py --tasks 30 --seeds 3 --wiki-mode live
```

### Run the cardiology study

```bash
python experiments/run_medical_study.py --tasks 200 --seeds 1 --wiki-mode live --output-dir experiments/results/medical_study_200tasks
```

### Launch the dashboard

```bash
streamlit run src/ui/app.py
```

## Configuration

LLM provider settings are in `src/llm/config.py`. API keys are loaded from environment variables:

| Variable | Description |
|----------|-------------|
| `ILAAS_API_KEY` | ILAAS platform API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OLLAMA_API_KEY` | Ollama API key (default: `ollama`) |

Set `ACTIVE_PROVIDER` in `src/llm/config.py` to switch between `ilaas`, `openai`, or `ollama`.
