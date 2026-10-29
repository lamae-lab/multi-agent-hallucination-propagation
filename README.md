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

## Citation

If you use this framework, please cite:

> Jabbar, S., Cruz, C., Bertolim, M.A. (2026). Hallucination propagation in multi-agent LLM systems: survival, distance, amplification, and mutation. *AGENTICS 2026*.

## References

- ChatMED-Project (2024). RAGCare-QA: a medical question answering dataset for retrieval-augmented generation. Hugging Face dataset.
- Cobbe, K. et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
- Cohen, R., Biran, M., Yoran, O., Globerson, A., Geva, M. (2023). LM vs LM: detecting factual errors via cross examination. arXiv:2305.13281.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.
- Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2024). RAGAS: automated evaluation of retrieval augmented generation. *EACL System Demonstrations*, 150–163.
- Guo, T. et al. (2024). Large language model based multi-agents: a survey of progress and challenges. *IJCAI*, 8048–8057.
- Hong, S. et al. (2024). MetaGPT: meta programming for a multi-agent collaborative framework. arXiv:2308.00352.
- Ji, Z. et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys* 55(12), 1–38.
- LangChain, Inc. (2024). LangGraph: build stateful multi-actor applications with LLMs. [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- Li, J., Cheng, X., Zhao, W.X., Nie, J.-Y., Wen, J.-R. (2023). HaluEval: a large-scale hallucination evaluation benchmark for large language models. *EMNLP*, 6449–6464.
- Manakul, P., Liusie, A., Gales, M.J.F. (2023). SelfCheckGPT: zero-resource black-box hallucination detection for generative large language models. *EMNLP*, 9004–9017.
- OpenAI (2025). OpenAI Agents SDK. [github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)
- Vosoughi, S., Roy, D., Aral, S. (2018). The spread of true and false news online. *Science* 359(6380), 1146–1151.
- Wu, Q. et al. (2023). AutoGen: enabling next-gen LLM applications via multi-agent conversation. arXiv:2308.08155.
- Yang, Z. et al. (2018). HotpotQA: a dataset for diverse, explainable multi-hop question answering. *EMNLP*, 2369–2380.
- Yao, S. et al. (2023). ReAct: synergizing reasoning and acting in language models. *ICLR*.
