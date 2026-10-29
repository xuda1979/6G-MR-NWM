import json
from dataclasses import replace

import pytest

from mr_nwm.control_records import (
    ExecutionFeedback,
    TransitionRecord,
    read_transitions,
    write_records,
)
from mr_nwm.slicing import SliceScenario, SliceState, TwoSliceQueueEnvironment, slice_utility


def test_round_robin_service_and_queue_conservation_are_hand_computed():
    env = TwoSliceQueueEnvironment(
        SliceScenario(total_blocks=2, arrivals=(0, 0, 0, 0), rates=(1, 2, 1, 1), alpha=0.5)
    )
    state = SliceState((3, 3, 3, 3), (1, 1, 1, 1))
    after = env.step(state, (2, 0, 0, 0))
    assert after.served == (1, 2, 0, 0)
    assert after.queues == (2, 1, 3, 3)
    assert after.average_rates == (1, 1.5, 0.5, 0.5)
    assert state.queues == (3, 3, 3, 3)
    assert sum(after.queues) + sum(after.served) == sum(state.queues)


def test_proportional_fair_score_uses_history_and_backlog_limit():
    env = TwoSliceQueueEnvironment(
        SliceScenario(total_blocks=2, arrivals=(0, 0, 0, 0), rates=(1, 2, 1, 1))
    )
    state = SliceState((3, 3, 3, 3), (1, 1, 1, 1))
    after = env.step(state, (2, 0, 1, 0))
    assert after.served == (0, 3, 0, 0)
    assert after.queues == (3, 0, 3, 3)
    assert all(x >= 0 for x in after.queues)


@pytest.mark.parametrize("action", [(2, 1, 0, 0), (0.5, 1.5, 0, 0), (1, 1, 2, 0)])
def test_environment_rejects_infeasible_resource_or_scheduler_actions(action):
    env = TwoSliceQueueEnvironment(SliceScenario(total_blocks=2))
    with pytest.raises(ValueError):
        env.step(SliceState((1, 1, 1, 1)), action)


def test_rollout_preserves_initial_state_and_honors_action_sequence():
    env = TwoSliceQueueEnvironment(SliceScenario(total_blocks=2, arrivals=(0, 0, 0, 0)))
    state = SliceState((2, 2, 2, 2))
    first = env.rollout(state, ((2, 0, 0, 0), (0, 2, 0, 0)))
    second = env.rollout(state, ((0, 2, 0, 0), (2, 0, 0, 0)))
    assert first[0].queues != second[0].queues
    assert first[-1].queues == second[-1].queues
    assert state.queues == (2, 2, 2, 2)
    assert len(env.candidates()) == 12


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_synthetic_state_and_scenario_reject_bad_numbers(bad):
    with pytest.raises(ValueError):
        SliceState((bad, 0, 0, 0))
    with pytest.raises(ValueError):
        SliceScenario(arrivals=(bad, 0, 0, 0))


def make_record(*, status="applied", applied=(1, 1, 0, 0), effective_at=1.0):
    feedback = ExecutionFeedback(
        episode_id="episode-a",
        request_id="action-1",
        requested_action=(2, 2, 0, 0),
        applied_action=applied,
        requested_at=1.0,
        effective_at=effective_at,
        status=status,
        source="simulated",
        duration_seconds=1.0,
    )
    return TransitionRecord(
        feedback=feedback,
        state=(2, 2, 2, 2),
        next_state=(1, 2, 1, 2),
        observed_at=0.9,
        available_at=1.0,
        next_observed_at=2.0,
        next_available_at=2.1,
        reward=2.0,
        group_id="scenario-a",
    )


def test_training_records_use_actual_applied_action_and_refuse_future_inputs(tmp_path):
    env = TwoSliceQueueEnvironment(SliceScenario(total_blocks=2))
    record = make_record()
    path = tmp_path / "transitions.jsonl"
    write_records(path, [record], env.spec)
    restored = read_transitions(path, env.spec)
    assert restored[0].feedback.applied_action == (1, 1, 0, 0)
    assert restored[0].feedback.applied_action != restored[0].feedback.requested_action
    with pytest.raises(FileExistsError):
        write_records(path, [record], env.spec)
    with pytest.raises(ValueError, match="available|causal"):
        replace(record, available_at=1.1)
    with pytest.raises(ValueError, match="effective|causal"):
        replace(record, feedback=replace(record.feedback, effective_at=2.5))
    with pytest.raises(ValueError, match="duration"):
        replace(record, feedback=replace(record.feedback, duration_seconds=0.5))


def test_rejected_requests_never_become_training_actions(tmp_path):
    env = TwoSliceQueueEnvironment(SliceScenario(total_blocks=2))
    rejected = ExecutionFeedback("e", "r", (1, 1, 0, 0), None, 0.0, None, "rejected")
    with pytest.raises(ValueError, match="applied"):
        replace(make_record(), feedback=rejected)
    with pytest.raises(ValueError):
        replace(rejected, applied_action=(0, 0, 0, 0))
    with pytest.raises(ValueError):
        write_records(tmp_path / "bad.jsonl", [make_record(applied=(2, 2, 0, 0))], env.spec)


def test_jsonl_tampering_and_duplicate_request_ids_are_detected(tmp_path):
    env = TwoSliceQueueEnvironment(SliceScenario(total_blocks=2))
    path = tmp_path / "data.jsonl"
    write_records(path, [make_record()], env.spec)
    raw = json.loads(path.read_text())
    raw["feedback"]["applied_action"] = None
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="applied"):
        read_transitions(path, env.spec)
    with pytest.raises(ValueError, match="duplicate"):
        write_records(path, [make_record(), make_record()], env.spec, overwrite=True)


def test_slice_utility_has_explicit_independently_checked_costs():
    state = SliceState((1, 2, 3, 4), served=(1, 2, 3, 4))
    assert slice_utility(state, (1, 1, 0, 0), queue_weight=0.5, block_cost=0.1) == 4.8
