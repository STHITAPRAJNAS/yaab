"""Run the coding-agent app offline: ``python -m cookbook.apps.coding_agent``."""

import asyncio

from . import run

print(asyncio.run(run()))
