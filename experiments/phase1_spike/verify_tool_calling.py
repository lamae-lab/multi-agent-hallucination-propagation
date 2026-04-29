"""Verify whether ILAAS models support OpenAI-format tool/function calling.

This is the critical gate for all framework spike tests.
If no model supports tool calling, all frameworks will fail at tool use.
"""

import json
import sys

import requests

sys.path.insert(0, ".")
from src.llm.config import PROVIDERS

ILAAS = PROVIDERS["ilaas"]

TOOL_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name.",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def test_tool_calling(model_name: str) -> dict:
    """Test if a model supports tool calling via the tools parameter."""
    url = f"{ILAAS['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {ILAAS['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": TOOL_DEFINITION,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 200,
        "stream": False,
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=60, verify=False
        )
        if response.status_code != 200:
            return {
                "model": model_name,
                "supports_tool_calling": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        data = response.json()
        message = data["choices"][0]["message"]
        has_tool_calls = "tool_calls" in message and message["tool_calls"] is not None

        return {
            "model": model_name,
            "supports_tool_calling": has_tool_calls,
            "tool_calls": message.get("tool_calls"),
            "content": message.get("content", ""),
            "raw_message": message,
        }
    except Exception as e:
        return {
            "model": model_name,
            "supports_tool_calling": False,
            "error": str(e),
        }


def main():
    # First, get all available models
    print("Fetching available ILAAS models...")
    headers = {
        "Authorization": f"Bearer {ILAAS['api_key']}",
        "Content-Type": "application/json",
    }
    resp = requests.get(f"{ILAAS['base_url']}/models", headers=headers, verify=False)
    models = [m["id"] for m in resp.json()["data"]]
    print(f"Available models: {models}\n")

    # Test each model
    results = []
    for model_name in models:
        print(f"Testing {model_name}...")
        result = test_tool_calling(model_name)
        results.append(result)

        if result["supports_tool_calling"]:
            print("  PASS - Tool calling supported!")
            if result.get("tool_calls"):
                print(f"  Tool calls: {json.dumps(result['tool_calls'], indent=2)}")
        else:
            error = result.get("error", "No tool_calls in response")
            content = result.get("content", "")[:100]
            print(f"  FAIL - {error}")
            if content:
                print(f"  Content: {content}")
        print()

    # Summary
    passing = [r for r in results if r["supports_tool_calling"]]
    print("=" * 60)
    print(f"SUMMARY: {len(passing)}/{len(results)} models support tool calling")
    if passing:
        print(f"Passing models: {[r['model'] for r in passing]}")
        print(f"\nRecommended model for spikes: {passing[0]['model']}")
    else:
        print("NO models support tool calling on ILAAS.")
        print("Decision: REIMPLEMENT ALL frameworks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
