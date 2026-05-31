"""Run this sample offline: `python -m samples.multi_agent_state`."""

import asyncio
import json

from . import run

print(json.dumps(asyncio.run(run()), indent=2))
