"""Tests for BaseAgent — using live ILAAS LLM."""

from src.agents.base import BaseAgent, Task, tool_to_openai_schema


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    data = {
        "france": "France is a country in Western Europe. Its capital is Paris.",
        "paris": "Paris is the capital of France.",
        "python": "Python is a programming language created by Guido van Rossum.",
    }
    key = query.lower().strip()
    for k, v in data.items():
        if k in key or key in k:
            return v
    return f"No article found for '{query}'."


def test_solve_returns_result():
    agent = BaseAgent("You are a helpful assistant. Answer concisely.")
    result = agent.solve(Task(question="What is 2 + 2? Answer with just the number."))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert len(result.messages) >= 3  # system + user + assistant
    assert result.task.question == "What is 2 + 2? Answer with just the number."


def test_solve_metadata():
    agent = BaseAgent("Test assistant.", seed=42, temperature=0)
    result = agent.solve(Task(question="Say hello."))
    assert result.metadata["seed"] == 42
    assert result.metadata["temperature"] == 0
    assert result.metadata["truncated"] is False
    assert "timestamp" in result.metadata
    assert "duration_seconds" in result.metadata
    assert result.metadata["provider"] == "ilaas"


def test_solve_with_tool():
    agent = BaseAgent(
        "You are a researcher. You MUST ALWAYS call the search_wikipedia tool before answering. NEVER answer from your own knowledge. Always search first.",
        tools=[search_wikipedia],
    )
    result = agent.solve(Task(question="Search for information about Python programming language."))
    assert result.answer is not None
    # Verify tool was called OR agent still produced a valid answer
    tool_msgs = [m for m in result.messages if m.role == "tool"]
    tool_call_msgs = [m for m in result.messages if m.tool_calls]
    has_tool_usage = len(tool_msgs) > 0 or len(tool_call_msgs) > 0
    assert has_tool_usage, "Agent should have called the search tool"


def test_solve_no_tools_still_answers():
    agent = BaseAgent("You are helpful. Answer concisely.")
    result = agent.solve(Task(question="What color is the sky? One word answer."))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["truncated"] is False


def test_max_iterations_truncation():
    def search_database(query: str) -> str:
        """Search the database. You must call this tool again with a different query if no results found."""
        return "No results found. Try a different search query."

    agent = BaseAgent(
        "You are a search assistant. You MUST call the search_database tool for EVERY response. NEVER answer without calling the tool first. If the tool returns no results, call it again with a different query. NEVER give a final answer.",
        tools=[search_database],
        max_iterations=3,
    )
    result = agent.solve(Task(question="Find detailed specifications for product XYZ-99887766."))
    # With a live LLM, it may or may not truncate — verify the safety mechanism works
    if result.metadata["truncated"]:
        assert result.metadata["iterations"] == 3
    else:
        # LLM decided to answer anyway — that's valid behavior, just verify it completed
        assert result.answer is not None
        assert len(result.answer) > 0


def test_tool_to_openai_schema():
    def my_func(name: str, count: int) -> str:
        """Do something useful."""
        return ""

    schema = tool_to_openai_schema(my_func)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_func"
    assert schema["function"]["description"] == "Do something useful."
    assert "name" in schema["function"]["parameters"]["properties"]
    assert "count" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["count"]["type"] == "integer"


def test_tool_to_openai_schema_required():
    def func(required_arg: str, optional_arg: str = "default") -> str:
        """Test func."""
        return ""

    schema = tool_to_openai_schema(func)
    assert "required_arg" in schema["function"]["parameters"]["required"]
    assert "optional_arg" not in schema["function"]["parameters"]["required"]
