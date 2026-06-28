"""Run the refund-pipeline app offline: ``python -m cookbook.apps.refund_pipeline``."""

import asyncio

from . import run

print(asyncio.run(run()))
