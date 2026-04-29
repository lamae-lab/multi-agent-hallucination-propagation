# Changelog

All notable methodology decisions and project changes are documented here.

## 2026-04-29

- **Project created.** Phase 0 — repository skeleton, dependency management, LLM provider configuration.
- **LLM provider architecture decision:** All three providers (Ollama, ILAAS, OpenAI/OpenRouter) use the OpenAI-compatible `/v1/chat/completions` endpoint, enabling a unified client with config-based provider switching.
- **Dependency choice:** Using `requests` for direct HTTP calls to LLM providers. `ollama` and `litellm` included for future phases.
