"""Behavior contracts for finite action spaces and bounded multistep planning."""

from dataclasses import dataclass

import pytest

from mr_nwm.action_space import ActionSpec
from mr_nwm.control_planner import BoundedPlanner, PlannerConfig


def scalar_spec(**overrides):
    return ActionSpec(
        **{
            "name": "test-command",
            "names": ("command",),
            "units": ("normalized",),
            "lower": (0.0,),
            "upper": (1.0,),
            **overrides,
        }
    )


def test_two_step_search_can_choose_a_non_greedy_first_action():
    # Action 1 invests now (-1) and unlocks a later reward of 5. Action 0
    # gives an immediate reward of 1 but no opportunity at the next step.
    def predict(initial, sequence):
        state = initial
        result = []
        for index, action in enumerate(sequence):
            if index == 0:
                state = {"invested": action[0], "reward": 1 - 2 * action[0]}
            else:
                state = {"invested": state["invested"], "reward": 5 * state["invested"]}
            result.append(state)
        return result

    def reward(state, action):
        return state["reward"]

    candidates = [(0,), (1,)]
    greedy = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1)).plan(
        {"invested": 0.0, "reward": 0.0}, candidates, predict, reward
    )
    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=2)).plan(
        {"invested": 0.0, "reward": 0.0}, candidates, predict, reward
    )
    assert greedy.first_action == (0.0,)
    assert result.status == "planned"
    assert result.first_action == (1.0,)
    assert result.objective == pytest.approx(4.0)
    assert result.considered == result.evaluated == 4
    assert not result.truncated


def test_discount_and_switching_cost_have_independently_computed_value():
    # Only one legal candidate. At t=0 switching from 0 to 1 costs 2;
    # later, the same action has no switching cost. 3 - 2 + 0.5 * 3 = 2.5.
    result = BoundedPlanner(
        scalar_spec(), PlannerConfig(horizon=2, discount=0.5, switch_cost=2.0)
    ).plan(0.0, [(1,)], lambda state, seq: [3.0, 3.0], lambda state, a: state, (0,))
    assert result.objective == pytest.approx(2.5)
    assert result.sequence == ((1.0,), (1.0,))


def test_switch_penalty_counts_action_changes_without_mixing_coordinate_units():
    spec = ActionSpec("mixed", ("power", "blocks"), ("W", "RB"), (0, 0), (10, 100))
    result = BoundedPlanner(spec, PlannerConfig(horizon=1, switch_cost=3)).plan(
        0.0, [(2, 50)], lambda state, seq: [8.0], lambda state, action: state, (1, 1)
    )
    assert result.objective == 5.0


def test_schema_enforces_integer_resource_budget_and_slew():
    spec = ActionSpec(
        "slice-budget",
        ("slice_0", "slice_1", "mode"),
        ("RB", "RB", "code"),
        (0, 0, 0),
        (6, 6, 1),
        integer_indices=(0, 1, 2),
        sum_indices=(0, 1),
        sum_limit=6,
        max_delta=(2, 2, 1),
    )
    spec.validate((4, 2, 1), (2, 4, 0))
    for action, previous, match in [
        ((4, 3, 0), (3, 3, 0), "sum|budget"),
        ((3.5, 2, 0), (3, 3, 0), "integer"),
        ((6, 0, 0), (3, 3, 0), "delta|slew"),
        ((3, 3, 0), None, "previous"),
        ((3, 3), (3, 3, 0), "shape|dimension"),
        ((3, 3, 2), (3, 3, 0), "bound"),
    ]:
        with pytest.raises(ValueError, match=match):
            spec.validate(action, previous)


@pytest.mark.parametrize("action", [(float("nan"),), (float("inf"),), (-0.1,), (1.1,)])
def test_action_validation_rejects_nonfinite_and_out_of_bounds(action):
    with pytest.raises(ValueError):
        scalar_spec().validate(action)


def test_disabled_capability_never_produces_an_action():
    spec = scalar_spec(enabled=False)
    with pytest.raises(ValueError, match="disabled|unsupported"):
        spec.validate((0.0,))
    result = BoundedPlanner(spec).plan(0.0, [(0,)], None, None)
    assert result.status == "no_plan"
    assert result.first_action is None


def test_masked_and_invalid_candidates_cannot_be_selected():
    planner = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1))
    result = planner.plan(
        0.0,
        [(0.2,), (0.9,), (float("nan"),), (100,)],
        lambda state, seq: [seq[0][0]],
        lambda state, action: state,
        candidate_mask=[True, False, True, True],
    )
    assert result.first_action == (0.2,)
    assert result.evaluated == 1
    with pytest.raises(ValueError, match="mask"):
        planner.plan(0.0, [(0,)], None, None, candidate_mask=[])
    with pytest.raises(ValueError, match="mask"):
        planner.plan(0.0, [(0,)], None, None, candidate_mask=[1])


def test_no_feasible_action_does_not_silently_create_zero_power():
    result = BoundedPlanner(scalar_spec()).plan(0.0, [(2,)], None, None)
    assert result.status == "no_plan"
    assert result.sequence == ()
    assert result.first_action is None
    assert result.objective is None
    assert result.evaluated == 0


def test_candidate_slew_is_checked_between_all_steps():
    spec = scalar_spec(max_delta=(0.4,))
    planner = BoundedPlanner(spec, PlannerConfig(horizon=2))
    result = planner.plan(
        0.0,
        [(0,), (0.4,), (0.8,)],
        lambda state, seq: [action[0] for action in seq],
        lambda state, action: state,
        previous_action=(0,),
    )
    assert result.sequence == ((0.4,), (0.8,))
    with pytest.raises(ValueError, match="previous"):
        planner.plan(0.0, [(0,)], None, None)


def test_search_budget_is_bounded_when_every_sequence_is_state_infeasible():
    calls = []
    planner = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=20, max_sequences=3))
    result = planner.plan(
        0.0,
        [(0,), (1,)],
        lambda state, seq: calls.append(seq) or [0.0] * len(seq),
        lambda state, action: 0.0,
        state_feasible=lambda state, action: False,
    )
    assert result.status == "no_plan"
    assert result.considered == result.evaluated == result.rejected == len(calls) == 3
    assert result.truncated


def test_search_budget_also_counts_slew_infeasible_sequences():
    planner = BoundedPlanner(
        scalar_spec(max_delta=(0.0,)), PlannerConfig(horizon=10, max_sequences=4)
    )
    result = planner.plan(
        0.0,
        [(0,), (1,)],
        lambda state, seq: [0.0] * len(seq),
        lambda state, action: 0.0,
        previous_action=(0,),
    )
    assert result.considered == 4
    assert result.evaluated == 1
    assert result.rejected == 3
    assert result.truncated


def test_candidate_count_bound_is_applied_before_model_calls():
    with pytest.raises(ValueError, match="candidate"):
        BoundedPlanner(scalar_spec(), PlannerConfig(max_candidates=2)).plan(
            0.0, [(0,), (0.2,), (0.4,)], None, None
        )


def test_malformed_action_iterables_are_not_consumed_without_a_dimension_bound():
    consumed = []

    def too_many_coordinates():
        for index in range(100):
            consumed.append(index)
            yield 0.0

    with pytest.raises(ValueError, match="dimension"):
        scalar_spec().validate(too_many_coordinates())
    assert len(consumed) == 2


def test_mutating_predictor_receives_an_independent_state_per_candidate():
    initial = {"history": [0.0]}
    starts = []

    def predict(state, seq):
        starts.append(state["history"].copy())
        state["history"].append(99.0)
        assert isinstance(seq, tuple) and isinstance(seq[0], tuple)
        return [seq[0][0]]

    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1)).plan(
        initial, [(0,), (1,)], predict, lambda state, action: state
    )
    assert result.first_action == (1.0,)
    assert starts == [[0.0], [0.0]]
    assert initial == {"history": [0.0]}


def test_forecast_generator_snapshots_each_yield_before_later_mutation():
    observed = []

    def predict(initial, sequence):
        state = {"value": 0.0}
        for index in range(len(sequence)):
            state["value"] = float(index + 1)
            yield state

    def reward(state, action):
        observed.append(state["value"])
        return state["value"]

    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=3)).plan(
        0.0, [(0,)], predict, reward
    )
    assert observed == [1.0, 2.0, 3.0]
    assert result.objective == 6.0


@dataclass(frozen=True)
class PredictedState:
    queues: tuple[float, ...]
    mode: str = "synthetic"


def test_numeric_dataclass_states_are_valid_but_nonfinite_fields_are_not():
    planner = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1))
    result = planner.plan(
        PredictedState((0.0,)),
        [(0,)],
        lambda state, seq: [PredictedState((1.0,))],
        lambda state, action: -sum(state.queues),
    )
    assert result.objective == -1.0
    with pytest.raises(ValueError, match="finite"):
        planner.plan(
            PredictedState((0.0,)),
            [(0,)],
            lambda state, seq: [PredictedState((float("nan"),))],
            lambda state, action: 0.0,
        )


@pytest.mark.parametrize("states", [None, [0.0], [float("nan"), 0.0], ["x", "y"]])
def test_malformed_or_nonfinite_forecasts_fail_closed(states):
    with pytest.raises(ValueError, match="forecast|predict|finite|numeric"):
        BoundedPlanner(scalar_spec(), PlannerConfig(horizon=2)).plan(
            0.0, [(0,)], lambda state, seq: states, lambda state, action: 0.0
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid", True])
def test_nonfinite_or_nonnumeric_reward_fails_closed(value):
    with pytest.raises(ValueError, match="reward|finite|numeric"):
        BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1)).plan(
            0.0, [(0,)], lambda state, seq: [0.0], lambda state, action: value
        )


def test_nonboolean_state_constraint_is_not_treated_as_feasible():
    with pytest.raises(ValueError, match="bool"):
        BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1)).plan(
            0.0,
            [(0,)],
            lambda state, seq: [0.0],
            lambda state, action: 0.0,
            state_feasible=lambda state, action: "yes",
        )


def test_deadline_expiry_during_a_callback_returns_no_executable_action(monkeypatch):
    from mr_nwm import control_planner

    clock = [0.0]
    monkeypatch.setattr(control_planner.time, "monotonic", lambda: clock[0])

    def predict(state, seq):
        clock[0] += 2.0
        return [1.0]

    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1, max_seconds=1.0)).plan(
        0.0, [(0,)], predict, lambda state, action: state
    )
    assert result.status == "no_plan"
    assert result.first_action is None
    assert result.truncated
    assert "deadline" in result.reason


def test_deadline_also_covers_reward_computation(monkeypatch):
    from mr_nwm import control_planner

    clock = [0.0]
    monkeypatch.setattr(control_planner.time, "monotonic", lambda: clock[0])

    def reward(state, action):
        clock[0] += 2.0
        return 1.0

    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1, max_seconds=1.0)).plan(
        0.0, [(0,)], lambda state, seq: [0.0], reward
    )
    assert result.status == "no_plan"
    assert result.first_action is None


def test_duplicate_actions_are_deduplicated_and_ties_keep_input_order():
    result = BoundedPlanner(scalar_spec(), PlannerConfig(horizon=1)).plan(
        0.0, [(1,), (0,), (1,)], lambda state, seq: [0.0], lambda state, action: 0.0
    )
    assert result.first_action == (1.0,)
    assert result.considered == result.evaluated == 2
    assert not result.truncated


@pytest.mark.parametrize(
    "overrides",
    [
        {"names": ("x", "x"), "units": ("u", "u"), "lower": (0, 0), "upper": (1, 1)},
        {"lower": (2,)},
        {"upper": (float("nan"),)},
        {"integer_indices": (1,)},
        {"sum_indices": (0,), "sum_limit": float("inf")},
        {"max_delta": (-1,)},
        {"enabled": "yes"},
    ],
)
def test_invalid_action_spec_fails_at_construction(overrides):
    with pytest.raises(ValueError):
        scalar_spec(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": True},
        {"max_sequences": 0},
        {"max_candidates": 0},
        {"discount": 1.1},
        {"switch_cost": -1},
        {"max_seconds": float("nan")},
    ],
)
def test_invalid_planner_config_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        PlannerConfig(**kwargs)
