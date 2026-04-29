# Changelog

All notable methodology decisions and project changes are documented here.

## 2026-04-29

- **Project created.** Phase 0 — repository skeleton, dependency management, LLM provider configuration.
- **LLM provider architecture decision:** All three providers (Ollama, ILAAS, OpenAI/OpenRouter) use the OpenAI-compatible `/v1/chat/completions` endpoint, enabling a unified client with config-based provider switching.
- **Dependency choice:** Using `requests` for direct HTTP calls to LLM providers. `ollama` and `litellm` included for future phases.
- **Phase 1 spike — framework decision (3/4 pass → use real frameworks):**
  - **LangGraph** → use real framework (3/5 correct, tool calling works, state-graph pattern visible)
  - **CrewAI** → reimplement (~200 LOC hub-and-spoke). LiteLLM layer causes API errors with ILAAS; only 1/5 correct.
  - **AutoGen/AG2** → use real framework via AG2 v0.7 API (3/5 correct, RoundRobinGroupChat works)
  - **OpenAI Agents SDK** → use real framework (4/5 correct, best result, handoffs work cleanly)
  - **Tool-calling models on ILAAS:** gpt-oss-120b, mistral-small-3.2-24b, qwen-3.6-35b-instruct (3/7 models)
  - **Default model changed** from llama-3.3-70b to qwen-3.6-35b-instruct (tool-calling support required)
  - Evidence: `experiments/phase1_spike/RESULTS.md`
