"""Deterministic two-slice fluid queues, not a calibrated radio/packet simulator.

Blocks and payload are abstract. RR rotates resource grants; PF uses the
rate/historical-throughput score. Neither mode is a 3GPP conformance claim.
"""

import math
from dataclasses import dataclass
from itertools import product

from .action_space import ActionSpec


def _vector(values, length, name, *, positive=False):
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a vector") from error
    if len(values) != length or any(isinstance(x, bool) for x in values):
        raise ValueError(f"{name} has wrong shape or type")
    try:
        values = tuple(float(x) for x in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if any(not math.isfinite(x) or x < 0 or (positive and x == 0) for x in values):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return values


@dataclass(frozen=True)
class SliceScenario:
    total_blocks: int = 4
    arrivals: tuple[float, ...] = (0.5, 0.8, 0.4, 0.6)
    rates: tuple[float, ...] = (1.0, 2.0, 1.5, 0.75)
    alpha: float = 0.2
    step_seconds: float = 1.0

    def __post_init__(self):
        if isinstance(self.total_blocks, bool) or not isinstance(self.total_blocks, int):
            raise ValueError("total_blocks must be an integer")
        if not 1 <= self.total_blocks <= 30:
            raise ValueError("total_blocks must be in [1, 30]")
        object.__setattr__(self, "arrivals", _vector(self.arrivals, 4, "arrivals"))
        object.__setattr__(self, "rates", _vector(self.rates, 4, "rates"))
        if not math.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(self.step_seconds) or self.step_seconds <= 0:
            raise ValueError("step_seconds must be positive and finite")


@dataclass(frozen=True)
class SliceState:
    queues: tuple[float, ...]
    average_rates: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    cursors: tuple[int, ...] = (0, 0)
    served: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "queues", _vector(self.queues, 4, "queues"))
        object.__setattr__(
            self, "average_rates", _vector(self.average_rates, 4, "average_rates", positive=True)
        )
        object.__setattr__(self, "served", _vector(self.served, 4, "served"))
        cursors = tuple(self.cursors)
        if len(cursors) != 2 or any(type(x) is not int or x not in (0, 1) for x in cursors):
            raise ValueError("cursors must be two integer UE offsets")
        object.__setattr__(self, "cursors", cursors)

    def features(self):
        return (
            self.queues + self.average_rates + tuple(float(x) for x in self.cursors) + self.served
        )


class TwoSliceQueueEnvironment:
    """Two slices, two UEs each, with arrivals at the start of each interval."""

    version = "two-slice-fluid-v1"

    def __init__(self, scenario=SliceScenario()):
        self.scenario = scenario
        self.spec = ActionSpec(
            name="two_slice_budget_scheduler_v1",
            names=("slice0_blocks", "slice1_blocks", "slice0_scheduler", "slice1_scheduler"),
            units=("abstract_blocks", "abstract_blocks", "rr0_pf1", "rr0_pf1"),
            lower=(0, 0, 0, 0),
            upper=(scenario.total_blocks, scenario.total_blocks, 1, 1),
            integer_indices=(0, 1, 2, 3),
            sum_indices=(0, 1),
            sum_limit=float(scenario.total_blocks),
        )

    def candidates(self):
        """All full-budget splits and scheduler combinations, in stable order."""
        return tuple(
            (float(first), float(self.scenario.total_blocks - first), float(a), float(b))
            for first in range(self.scenario.total_blocks + 1)
            for a, b in product((0, 1), repeat=2)
        )

    def step(self, state, action):
        self.spec.validate(action)
        if not isinstance(state, SliceState):
            raise ValueError("state must be a SliceState")
        c = self.scenario
        remaining = [q + a * c.step_seconds for q, a in zip(state.queues, c.arrivals)]
        served, cursors = [0.0] * 4, list(state.cursors)
        for slice_id in range(2):
            users = (2 * slice_id, 2 * slice_id + 1)
            for _ in range(int(action[slice_id])):
                if int(action[2 + slice_id]) == 0:
                    order = (users[cursors[slice_id]], users[1 - cursors[slice_id]])
                    active = [u for u in order if remaining[u] > 0]
                else:
                    active = sorted(
                        (u for u in users if remaining[u] > 0),
                        key=lambda u: (-c.rates[u] / state.average_rates[u], u),
                    )
                if not active:
                    break
                user = active[0]
                amount = min(remaining[user], c.rates[user] * c.step_seconds)
                remaining[user] -= amount
                served[user] += amount
                if int(action[2 + slice_id]) == 0:
                    cursors[slice_id] = 1 - (user - users[0])
        averages = tuple(
            max((1 - c.alpha) * old + c.alpha * used / c.step_seconds, 1e-9)
            for old, used in zip(state.average_rates, served)
        )
        return SliceState(tuple(remaining), averages, tuple(cursors), tuple(served))

    def rollout(self, initial, actions):
        state, states = initial, []
        for action in actions:
            state = self.step(state, action)
            states.append(state)
        return tuple(states)


def slice_utility(state, action, *, queue_weight=0.5, block_cost=0.01):
    """A new synthetic utility; it does not alter the published power benchmark."""
    if any(not math.isfinite(x) or x < 0 for x in (queue_weight, block_cost)):
        raise ValueError("utility costs must be finite and nonnegative")
    return sum(state.served) - queue_weight * sum(state.queues) - block_cost * sum(action[:2])
