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

1. Boiko, D.A., MacKnight, R., Kline, B., Gomes, G. (2023). Emergent autonomous scientific research capabilities of large language models. arXiv:2304.05332.
2. Chan, C.M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., Liu, Z. (2024). ChatEval: towards better LLM-based evaluators through multi-agent debate. *ICLR*.
3. Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
4. Cohen, R., Biran, M., Yoran, O., Globerson, A., Geva, M. (2023). LM vs LM: detecting factual errors via cross examination. arXiv:2305.13281.
5. CrewAI, Inc. (2024). CrewAI: framework for orchestrating role-playing autonomous AI agents. [github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)
6. Dobreva, J., Karasmanakis, I., Ivanisevic, F., Horvat, T., Kitanovski, D., Gams, M., Mishev, K., Misheva, M.S. (2025). RAGCare-QA: a benchmark dataset for evaluating retrieval-augmented generation pipelines in theoretical medical knowledge. *Data in Brief*, 112146.
7. Du, Y., Li, S., Torralba, A., Tenenbaum, J.B., Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. arXiv:2305.14325.
8. Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2024). RAGAS: automated evaluation of retrieval augmented generation. *EACL System Demonstrations*, 150–163.
9. Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N.V., Wiest, O., Zhang, X. (2024). Large language model based multi-agents: a survey of progress and challenges. *IJCAI*, 8048–8057.
10. Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., Wang, J., Wang, Z., Yau, S.K.S., Lin, Z., et al. (2024). MetaGPT: meta programming for a multi-agent collaborative framework. arXiv:2308.00352.
11. Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., et al. (2025). A survey on hallucination in large language models: principles, taxonomy, challenges, and open questions. *ACM Transactions on Information Systems*.
12. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A., Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys* 55(12), 1–38.
13. Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
14. Li, J., Cheng, X., Zhao, W.X., Nie, J.Y., Wen, J.R. (2023). HaluEval: a large-scale hallucination evaluation benchmark for large language models. *EMNLP*, 6449–6464.
15. Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., Shi, S. (2023). Encouraging divergent thinking in large language models through multi-agent debate. arXiv:2305.19118.
16. Lin, S., Hilton, J., Evans, O. (2022). TruthfulQA: measuring how models mimic human falsehoods. *ACL (Volume 1: Long Papers)*, 3214–3252.
17. Manakul, P., Liusie, A., Gales, M.J. (2023). SelfCheckGPT: zero-resource black-box hallucination detection for generative large language models. *EMNLP*, 9004–9017.
18. Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.t., Koh, P.W., Iyyer, M., Zettlemoyer, L., Hajishirzi, H. (2023). FActScore: fine-grained atomic evaluation of factual precision in long form text generation. *EMNLP*, 12076–12100.
19. OpenAI (2025). OpenAI Agents SDK. [github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)
20. Park, J.S., O'Brien, J., Cai, C.J., Morris, M.R., Liang, P., Bernstein, M.S. (2023). Generative agents: interactive simulacra of human behavior. *UIST*, 1–22.
21. Qian, C., Liu, W., Liu, H., Chen, N., Dang, Y., Li, J., Yang, C., Chen, W., Su, Y., Cong, X., et al. (2024). ChatDev: communicative agents for software development. *ACL (Volume 1: Long Papers)*, 15174–15186.
22. Vosoughi, S., Roy, D., Aral, S. (2018). The spread of true and false news online. *Science* 359(6380), 1146–1151.
23. Wang, J., Duan, Z. (2024). Agent AI with LangGraph: a modular framework for enhancing machine translation using large language models. arXiv:2412.03801.
24. Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., et al. (2023). AutoGen: enabling next-gen LLM applications via multi-agent conversation. arXiv:2308.08155.
25. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning, C.D. (2018). HotpotQA: a dataset for diverse, explainable multi-hop question answering. *EMNLP*, 2369–2380.
26. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y. (2023). ReAct: synergizing reasoning and acting in language models. *ICLR*.
27. Zhang, Y., Li, Y., Cui, L., Cai, D., Liu, L., Fu, T., Huang, X., Zhao, E., Zhang, Y., Chen, Y., et al. (2023). Siren's song in the AI ocean: a survey on hallucination in large language models. arXiv:2309.01219.
28. Zheng, L., Chiang, W.L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS*, vol. 36.
