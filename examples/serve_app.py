"""A minimal servable agent for the Dockerfile / `yaab serve` demo.

    yaab serve examples.serve_app:agent
    # or, with a real model:  YAAB_AGENT=examples.serve_app:agent

Swap TestModel for "openai/gpt-4o" (and set OPENAI_API_KEY) for real inference.
"""

from yaab import Agent
from yaab.testing import TestModel

agent = Agent(
    "assistant",
    model=TestModel("Hello from a served YAAB agent."),
    instructions="You are a helpful assistant.",
    registry_id="assistant",
)
