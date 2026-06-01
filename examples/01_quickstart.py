"""Quickstart: the three-line agent, tools, and typed structured output.

Runs fully offline with TestModel (no API key). Swap TestModel for a model
string like "openai/gpt-4o" to talk to a real provider.
"""

from pydantic import BaseModel

from yaab import Agent, tool
from yaab.testing import TestModel


# Tools are typed functions; the schema is auto-generated from the signature.
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# Typed structured output (validated, with reflection/retry).
class Weather(BaseModel):
    city: str
    temp_c: int


def main() -> dict:
    """Run the three quickstart patterns and return their results."""
    # 1) Three-line agent ------------------------------------------------
    agent = Agent("assistant", model=TestModel("Hello! How can I help?"))
    simple = agent.run_sync("hi").output
    print("simple:", simple)

    # 2) Tools (typed; schema auto-generated from the signature) ---------
    calc = Agent(
        "calc", model=TestModel(custom_output="The sum is 5", call_tools=["add"]), tools=[add]
    )
    with_tool = calc.run_sync("add 2 and 3").output
    print("with tool:", with_tool)

    # 3) Typed structured output (validated, with reflection/retry) ------
    weather = Agent(
        "weather",
        model=TestModel(structured_output={"city": "Paris", "temp_c": 21}),
        output_type=Weather,
    )
    structured = weather.run_sync("weather in Paris").output
    print("structured:", structured, "-> city =", structured.city)

    return {"simple": simple, "with_tool": with_tool, "structured": structured}


if __name__ == "__main__":
    main()
