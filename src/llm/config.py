"""LLM Provider Configuration.

Switch the active provider by changing ACTIVE_PROVIDER below.
All providers use the OpenAI-compatible /v1/chat/completions endpoint.

API keys are loaded from environment variables (or .env file).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Change this single variable to switch providers: "ollama", "ilaas", "openai"
# ---------------------------------------------------------------------------
ACTIVE_PROVIDER = "ilaas"

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------
PROVIDERS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://192.168.50.29:11434/v1"),
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        "model": "qwen2.5:14b",
        "timeout": 120,
    },
    "ilaas": {
        "base_url": os.getenv("ILAAS_BASE_URL", "https://llm.ilaas.fr/v1"),
        "api_key": os.getenv("ILAAS_API_KEY", ""),
        "model": "gpt-oss-120b",
        "timeout": 120,
        "verify_ssl": False,  # ILAAS cert revocation check fails on Windows
    },
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-4o",
        "timeout": 120,
    },
}
