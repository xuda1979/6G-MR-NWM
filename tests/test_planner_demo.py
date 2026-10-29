import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from mr_nwm.control_records import read_transitions
from mr_nwm.slicing import SliceScenario, TwoSliceQueueEnvironment


def test_demo_exports_replayable_applied_actions_and_authenticates_sources(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/planner_demo.py"),
        "--output",
        str(tmp_path / "smoke"),
        "--steps",
        "2",
        "--seed",
        "11",
    ]
    completed = subprocess.run(
        command, cwd=root, check=True, capture_output=True, text=True, timeout=60
    )
    assert json.loads(completed.stdout)["evidence"] == "simulated_only"
    summary = json.loads((tmp_path / "smoke/summary.json").read_text())
    env = TwoSliceQueueEnvironment(SliceScenario(**summary["scenario"]))
    records = read_transitions(tmp_path / "smoke/transitions.jsonl", env.spec)
    assert summary["transition_rows"] == 24 == len(records)
    assert len({r.group_id for r in records}) == 1
    assert all(r.feedback.source == "simulated" for r in records)
    coverage = [r for r in records if r.feedback.episode_id.endswith("/coverage")]
    assert {r.feedback.applied_action for r in coverage} == set(env.candidates())
    assert summary["reference_probe"]["plan"]["status"] == "planned"
    assert summary["reference_probe"]["field_control_evidence"] is False
    assert summary["policies"][1]["decisions"][0]["considered"] == 400
    assert not summary["policies"][1]["decisions"][0]["truncated"]

    # Independently recompute conservation/reward from serialized numeric fields.
    arrivals = summary["scenario"]["arrivals"]
    seconds = summary["scenario"]["step_seconds"]
    for record in records:
        queues, next_queues, served = (
            record.state[:4],
            record.next_state[:4],
            record.next_state[-4:],
        )
        for old, arrival, remaining, used in zip(queues, arrivals, next_queues, served):
            assert remaining + used == pytest.approx(old + arrival * seconds)
        expected = (
            sum(served) - 0.5 * sum(next_queues) - 0.01 * sum(record.feedback.applied_action[:2])
        )
        assert math.isfinite(expected) and record.reward == pytest.approx(expected)

    import hashlib

    for path, digest in summary["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256((tmp_path / "smoke/transitions.jsonl").read_bytes()).hexdigest()
        == (summary["transitions_sha256"])
    )
    second = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
    assert second.returncode != 0 and "output exists" in second.stderr
