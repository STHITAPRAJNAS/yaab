"""Recipe: record/replay — deterministic offline tests of real model behaviour.

``CassetteModel`` records a real model's responses once, then replays them with
no key or network. This recipe records (against a scripted ``TestModel`` standing
in for a real one) then replays from the same cassette.

    python -m cookbook.recipes.record_replay
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cookbook._harness import expect
from yaab import Agent
from yaab.models.test_model import TestModel
from yaab.testing import CassetteModel


async def run() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "session.json"

        # 1. Record: wrap the (here, scripted) model and capture its answer.
        recorder = CassetteModel(path, inner=TestModel(custom_output="Paris."), mode="record")
        rec_agent: Agent = Agent("geo", model=recorder, instructions="One word.")
        recorded = await rec_agent.run("Capital of France?")
        expect(path.exists(), "recording should have written a cassette")

        # 2. Replay: no inner model, no network — the saved answer comes back.
        replay = CassetteModel(path, mode="replay")
        play_agent: Agent = Agent("geo", model=replay, instructions="One word.")
        replayed = await play_agent.run("Capital of France?")

    expect(
        str(recorded.output) == str(replayed.output),
        f"replay should match the recording: {recorded.output!r} vs {replayed.output!r}",
    )
    return {"recorded": str(recorded.output), "replayed": str(replayed.output)}


if __name__ == "__main__":
    print(asyncio.run(run()))
