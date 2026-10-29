"""Explicit, immutable numeric actuator contracts, independent of a simulator."""

import math
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from typing import Iterable

Action = tuple[float, ...]


def _numeric_action(values: Iterable[float], dimension: int, label: str = "action") -> Action:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric vector")
    try:
        values = tuple(islice(iter(values), dimension + 1))
    except TypeError as exc:
        raise ValueError(f"{label} must be a numeric vector") from exc
    if len(values) != dimension:
        raise ValueError(f"{label} dimension must be {dimension}")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{label} must contain finite numeric values")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{label} must contain finite numeric values") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain finite numeric values")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class ActionSpec:
    """A versionable capability's numeric bounds, units, and coupled limits.

    An empty action set is not repaired by inventing an implicit zero action.
    A previous action is a finite physical reference for slew checking; it need
    not satisfy newly tightened current bounds. Boundaries are checked without
    clipping or silently rounding requested actuator values.
    """

    name: str
    names: tuple[str, ...]
    units: tuple[str, ...]
    lower: Action
    upper: Action
    integer_indices: tuple[int, ...] = ()
    sum_indices: tuple[int, ...] = ()
    sum_limit: float | None = None
    max_delta: Action | None = None
    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("capability name must be nonempty")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be bool")
        for field in ("names", "units"):
            values = getattr(self, field)
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{field} must be a sequence of nonempty strings")
            try:
                values = tuple(values)
            except TypeError as exc:
                raise ValueError(f"{field} must be a sequence") from exc
            if not values or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{field} must contain nonempty strings")
            object.__setattr__(self, field, values)
        dimension = len(self.names)
        if len(set(self.names)) != dimension or len(self.units) != dimension:
            raise ValueError("coordinate names must be unique and units must match dimensions")
        for field in ("lower", "upper", "max_delta"):
            values = getattr(self, field)
            if values is None and field != "max_delta":
                raise ValueError(f"{field} bounds are required")
            if values is not None:
                object.__setattr__(self, field, _numeric_action(values, dimension, field))
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("lower bounds cannot exceed upper bounds")
        for field in ("integer_indices", "sum_indices"):
            try:
                indices = tuple(getattr(self, field))
            except TypeError as exc:
                raise ValueError(f"{field} must contain coordinate indices") from exc
            if any(type(index) is not int or not 0 <= index < dimension for index in indices):
                raise ValueError(f"{field} contains an invalid coordinate index")
            if len(set(indices)) != len(indices):
                raise ValueError(f"{field} contains duplicate coordinate indices")
            object.__setattr__(self, field, indices)
        if bool(self.sum_indices) != (self.sum_limit is not None):
            raise ValueError("sum_indices and sum_limit must be provided together")
        if self.sum_limit is not None:
            limit = _numeric_action((self.sum_limit,), 1, "sum_limit")[0]
            object.__setattr__(self, "sum_limit", limit)
        if self.max_delta is not None and any(delta < 0 for delta in self.max_delta):
            raise ValueError("max_delta must be nonnegative")

    def validate(self, action: Iterable[float], previous: Iterable[float] | None = None) -> None:
        if not self.enabled:
            raise ValueError(f"unsupported or disabled capability: {self.name}")
        action = _numeric_action(action, len(self.names))
        if any(not lo <= value <= hi for value, lo, hi in zip(action, self.lower, self.upper)):
            raise ValueError("action violates coordinate bounds")
        if any(not action[index].is_integer() for index in self.integer_indices):
            raise ValueError("action violates integer coordinate requirements")
        if self.sum_limit is not None:
            try:
                total = math.fsum(action[index] for index in self.sum_indices)
            except OverflowError as exc:
                raise ValueError("action sum budget must remain finite") from exc
            if total > self.sum_limit:
                raise ValueError("action violates sum budget")
        if previous is not None:
            previous = _numeric_action(previous, len(self.names), "previous action")
        if self.max_delta is not None:
            if previous is None:
                raise ValueError("previous action is required to check slew limits")
            if any(
                abs(value - old) > delta
                for value, old, delta in zip(action, previous, self.max_delta)
            ):
                raise ValueError("action violates max_delta slew limits")
