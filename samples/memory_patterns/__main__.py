"""Run this sample offline: `python -m samples.memory_patterns`."""

import asyncio
import json

from . import run

print(json.dumps(asyncio.run(run()), indent=2))
