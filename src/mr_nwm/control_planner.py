"""Bounded finite-horizon search over explicit, independently predicted actions.

This is an offline/research planning primitive, not a network execution service.
A synchronous predictor cannot be preempted. An expired deadline yields no
action after the callback returns, even if an earlier sequence was feasible.
"""

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from itertools import islice, product
from numbers import Real
from typing import Any, Callable, Iterable

from .action_space import Action, ActionSpec, _numeric_action


def _finite_scalar(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric scalar") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _finite_state(value, label: str = "forecast", depth: int = 0) -> int:
    """Check numeric leaves without forcing callers into one state container.

    String metadata and booleans are allowed inside state records. A state
    must still contain at least one numeric leaf. Cyclic/too-deep records and
    opaque objects fail explicitly; dataclasses and tolist-compatible arrays
    are supported without importing an array framework here.
    """
    if depth > 64:
        raise ValueError(f"{label} is cyclic or too deeply nested")
    if isinstance(value, bool) or isinstance(value, str):
        return 0
    if isinstance(value, Real):
        _finite_scalar(value, label)
        return 1
    if is_dataclass(value) and not isinstance(value, type):
        children = [getattr(value, field.name) for field in fields(value)]
    elif isinstance(value, Mapping):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    elif hasattr(value, "tolist") and callable(value.tolist):
        return _finite_state(value.tolist(), label, depth + 1)
    else:
        raise ValueError(f"{label} must contain finite numeric state values")
    return sum(_finite_state(child, label, depth + 1) for child in children)


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 2
    max_sequences: int = 512
    discount: float = 1.0
    switch_cost: float = 0.0
    max_seconds: float | None = None
    max_candidates: int = 128

    def __post_init__(self):
        for name in ("horizon", "max_sequences", "max_candidates"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon > 64:
            raise ValueError("horizon exceeds the bounded planner limit of 64")
        discount = _finite_scalar(self.discount, "discount")
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        switch_cost = _finite_scalar(self.switch_cost, "switch_cost")
        if switch_cost < 0:
            raise ValueError("switch_cost must be nonnegative")
        if self.max_seconds is not None and _finite_scalar(self.max_seconds, "max_seconds") <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass(frozen=True)
class PlanResult:
    status: str
    sequence: tuple[Action, ...]
    objective: float | None
    evaluated: int
    considered: int
    rejected: int
    truncated: bool
    reason: str

    @property
    def first_action(self) -> Action | None:
        return self.sequence[0] if self.sequence else None


class BoundedPlanner:
    """Stable, finite enumeration with capability, budget, and slew checks.

    ``considered`` counts complete sequences, including rejected ones, so a
    constraint cannot turn the configured search budget into an unbounded scan.
    ``evaluated`` counts calls to the predictor; ``rejected`` counts sequences
    rejected for slew or predicted-state infeasibility. Invalid or masked input
    candidates are filtered before enumeration and are not sequence counts.
    Exact maximization is claimed only over an exhaustive, non-truncated search
    of the supplied candidate set under the given predictor/reward callbacks.
    Predictors must keep candidate memory in the supplied state or reset it on
    each call; arbitrary hidden state inside an external callable is not cloned.
    """

    def __init__(self, spec: ActionSpec, config: PlannerConfig | None = None):
        if not isinstance(spec, ActionSpec):
            raise ValueError("spec must be an ActionSpec")
        if config is not None and not isinstance(config, PlannerConfig):
            raise ValueError("config must be a PlannerConfig")
        self.spec = spec
        self.config = config or PlannerConfig()

    def plan(
        self,
        state: Any,
        candidates: Iterable[Action],
        predict: Callable,
        reward: Callable,
        previous_action: Action | None = None,
        candidate_mask: Iterable[bool] | None = None,
        state_feasible: Callable | None = None,
    ) -> PlanResult:
        config, spec = self.config, self.spec
        started = time.monotonic()
        deadline = None if config.max_seconds is None else started + config.max_seconds
        considered = evaluated = rejected = 0

        def no_plan(reason, truncated=False):
            return PlanResult(
                "no_plan", (), None, evaluated, considered, rejected, truncated, reason
            )

        def expired():
            return deadline is not None and time.monotonic() >= deadline

        if not spec.enabled:
            return no_plan("unsupported or disabled capability")
        if spec.max_delta is not None and previous_action is None:
            raise ValueError("previous action is required to check slew limits")
        if previous_action is not None:
            previous_action = _numeric_action(previous_action, len(spec.names), "previous action")
        if not _finite_state(state, "initial state"):
            raise ValueError("initial state must contain finite numeric values")
        try:
            raw = tuple(islice(iter(candidates), config.max_candidates + 1))
        except TypeError as exc:
            raise ValueError("candidates must be an iterable of action vectors") from exc
        if len(raw) > config.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        if candidate_mask is None:
            mask = (True,) * len(raw)
        else:
            try:
                mask = tuple(islice(iter(candidate_mask), len(raw) + 1))
            except TypeError as exc:
                raise ValueError("candidate_mask must be an iterable of bool") from exc
            if len(mask) != len(raw) or any(type(value) is not bool for value in mask):
                raise ValueError("candidate_mask must contain one bool per candidate")
        valid = []
        seen = set()
        for action, enabled in zip(raw, mask):
            if not enabled:
                continue
            try:
                action = _numeric_action(action, len(spec.names))
                # Slew is checked for the full sequence below. First-step-illegal
                # actions can remain useful as later sequence entries.
                reference = action if spec.max_delta is not None else None
                spec.validate(action, reference)
            except ValueError:
                continue
            if action not in seen:
                valid.append(action)
                seen.add(action)
        if not valid:
            return no_plan("no valid unmasked candidates")
        if not callable(predict) or not callable(reward):
            raise ValueError("predict and reward must be callable")
        if state_feasible is not None and not callable(state_feasible):
            raise ValueError("state_feasible must be callable")
        total = len(valid) ** config.horizon
        truncated = total > config.max_sequences
        best_sequence = ()
        best_value = None
        for sequence in islice(product(valid, repeat=config.horizon), config.max_sequences):
            if expired():
                return no_plan("planning deadline exceeded", True)
            considered += 1
            old = previous_action
            try:
                for action in sequence:
                    spec.validate(action, old)
                    old = action
            except ValueError:
                rejected += 1
                continue
            evaluated += 1
            forecast = predict(copy.deepcopy(state), sequence)
            if expired():
                return no_plan("planning deadline exceeded after predictor callback", True)
            if isinstance(forecast, (str, bytes, Mapping)):
                raise ValueError("predict forecast must contain exactly horizon subsequent states")
            try:
                # Snapshot each yielded state before requesting the next one:
                # generators may advance and reuse the same mutable buffer.
                forecast = tuple(
                    copy.deepcopy(next_state)
                    for next_state in islice(iter(forecast), config.horizon + 1)
                )
            except TypeError as exc:
                raise ValueError(
                    "predict forecast must be an iterable of subsequent states"
                ) from exc
            if len(forecast) != config.horizon:
                raise ValueError("predict forecast must contain exactly horizon subsequent states")
            for next_state in forecast:
                if not _finite_state(next_state):
                    raise ValueError("predict forecast states must contain numeric values")
            if state_feasible is not None:
                feasible = True
                for next_state, action in zip(forecast, sequence):
                    verdict = state_feasible(copy.deepcopy(next_state), action)
                    if type(verdict) is not bool:
                        raise ValueError("state_feasible must return bool")
                    feasible = feasible and verdict
                if not feasible:
                    rejected += 1
                    if expired():
                        return no_plan("planning deadline exceeded after constraint callback", True)
                    continue
            value = 0.0
            old = previous_action
            for index, (next_state, action) in enumerate(zip(forecast, sequence)):
                step_value = _finite_scalar(reward(copy.deepcopy(next_state), action), "reward")
                if old is not None and action != old:
                    step_value -= config.switch_cost
                value += config.discount**index * step_value
                if not math.isfinite(value):
                    raise ValueError("discounted reward objective must remain finite")
                old = action
            if expired():
                return no_plan("planning deadline exceeded after scoring callback", True)
            if best_value is None or value > best_value:
                best_sequence, best_value = sequence, value
        if expired():
            return no_plan("planning deadline exceeded", True)
        if best_value is None:
            return no_plan("no feasible candidate sequence", truncated)
        reason = (
            "sequence budget exhausted" if truncated else "exhaustive supplied-candidate search"
        )
        return PlanResult(
            "planned", best_sequence, best_value, evaluated, considered, rejected, truncated, reason
        )
