"""Run the research-assistant app offline: ``python -m cookbook.apps.research_assistant``."""

import asyncio

from . import run

print(asyncio.run(run()))
