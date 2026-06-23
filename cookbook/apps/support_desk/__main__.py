"""Run the support-desk app offline: ``python -m cookbook.apps.support_desk``."""

import asyncio

from . import run

print(asyncio.run(run()))
