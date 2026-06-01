"""Quickstart: the three-line agent, tools, and typed structured output.

Runs fully offline with TestModel (no API key). Swap TestModel for a model
string like "openai/gpt-4o" to talk to a real provider.
"""

from pydantic import BaseModel

from yaab import Agent, tool
from yaab.testing import TestModel

# 1) Three-line agent ---------------------------------------------------
agent = Agent("assistant", model=TestModel("Hello! How can I help?"))
print("simple:", agent.run_sync("hi").output)


# 2) Tools (typed; schema auto-generated from the signature) ------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


calc = Agent("calc", model=TestModel(custom_output="The sum is 5", call_tools=["add"]), tools=[add])
print("with tool:", calc.run_sync("add 2 and 3").output)


# 3) Typed structured output (validated, with reflection/retry) ---------
class Weather(BaseModel):
    city: str
    temp_c: int


weather = Agent(
    "weather",
    model=TestModel(structured_output={"city": "Paris", "temp_c": 21}),
    output_type=Weather,
)
result = weather.run_sync("weather in Paris")
print("structured:", result.output, "-> city =", result.output.city)
