# Phase 1 Spike Test Results

**Date**: 2026-04-29
**LLM Provider**: ILAAS (llm.ilaas.fr)
**Model**: qwen-3.6-35b-instruct (selected after tool-calling verification)

## Tool-Calling Verification

| Model | Supports Tool Calling | Notes |
|-------|----------------------|-------|
| gemma-4-31b | No | API error (no choices in response) |
| gpt-oss-120b | Yes | Correct tool call with proper arguments |
| llama-3.1-8b | No | API error (no choices in response) |
| llama-3.3-70b | No | API error (no choices in response) |
| mistral-medium-250523 | No | Returns tool call as text, not structured tool_calls field |
| mistral-small-3.2-24b | Yes | Correct tool call with proper arguments |
| qwen-3.6-35b-instruct | Yes | Correct tool call with proper arguments |

**3/7 models support tool calling.** Using qwen-3.6-35b-instruct for spike tests (good balance of capability and speed).

## Framework Results

| Framework | Ran Successfully | Tool-Call Rate | Trace Quality | Correct Answers | Failure Mode | Decision |
|-----------|-----------------|----------------|---------------|-----------------|--------------|----------|
| LangGraph | Yes | 7/10 (70%) | Good — tool calls and state visible | 3/5 | None | **Use real** |
| CrewAI | Yes (partial) | Low — API errors on most questions | Partial — verbose output but errors | 1/5 | LiteLLM/ILAAS API errors on tool calls | **Reimplement** |
| AutoGen/AG2 | Yes | Good — tools called via RoundRobin | Good — message history visible | 3/5 | None | **Use real** |
| OpenAI Agents SDK | Yes | Good — tools + handoffs working | Good — agent handoffs visible | 4/5 | None | **Use real** |

## Decision Rule

From project plan:
- **>=3 pass**: Use real frameworks for those, reimplement failures
- **<=2 pass**: Mixed approach with fidelity validation
- **0 pass**: Escalate to larger model; reimplementation is last resort

## Decision Summary

**3 of 4 frameworks passed** → use real frameworks for those, reimplement CrewAI.

| Framework | Decision | Rationale |
|-----------|----------|-----------|
| LangGraph | **Use real framework** | Ran successfully, 70% tool-call rate, 3/5 correct. State-graph pattern visible in traces. Minor deprecation warning (moved to `langchain.agents`). |
| CrewAI | **Reimplement** | Ran but with significant API errors through LiteLLM layer. Only 1/5 correct. The LiteLLM abstraction adds friction with non-standard providers. Hierarchical crew pattern is simple enough to reimplement (~200 LOC). |
| AutoGen/AG2 | **Use real framework** | Ran successfully with AG2 v0.7 API. 3/5 correct. RoundRobinGroupChat with tool-calling agents works. Shared conversation history pattern clearly visible. |
| OpenAI Agents SDK | **Use real framework** | Best result: 4/5 correct. Tool calling and handoff patterns work cleanly. Agent handoffs visible in output. Requires `set_tracing_disabled(True)` for non-OpenAI providers. |

**Key findings**:
- ILAAS model `qwen-3.6-35b-instruct` supports OpenAI-format tool calling reliably.
- 3 models on ILAAS support tool calling: gpt-oss-120b, mistral-small-3.2-24b, qwen-3.6-35b-instruct.
- CrewAI's LiteLLM dependency causes issues with non-standard providers; reimplementing the hub-and-spoke pattern directly is preferred.
- All q3 answers got TD Garden's opening year wrong (1995 vs 1999) — this is a mock data issue, not a framework issue.
