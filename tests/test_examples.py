"""The recorded runs in examples/ come from real API calls (the eval). Replaying them guards the loop against
changes that would alter what the model sees or what stops at the checkpoint."""
import json
from pathlib import Path

import pytest

from ops_agent import agent

FILES = sorted((Path(__file__).parents[1] / "examples").glob("*.jsonl"))


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_recorded_run_replays_identically(path):
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    _, diffs = agent.replay(lines[0]["task"], lines[1:], lines[0]["status"])
    assert diffs == []


def test_there_are_examples():
    assert len(FILES) >= 3
