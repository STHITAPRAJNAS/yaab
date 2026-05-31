"""Run this sample offline: `python -m samples.personal_assistant`."""

import asyncio
import json

from . import run

print(json.dumps(asyncio.run(run()), indent=2))
