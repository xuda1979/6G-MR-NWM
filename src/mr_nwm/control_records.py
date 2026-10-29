"""Causal, applied-action transition records for supervised dynamics datasets."""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import islice
from numbers import Real
from pathlib import Path


def _action(values, name, dimension=None):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a finite numeric vector")
    try:
        values = tuple(islice(iter(values), 65))
    except TypeError as error:
        raise ValueError(f"{name} must be a vector") from error
    if not 1 <= len(values) <= 64 or (dimension is not None and len(values) != dimension):
        raise ValueError(f"{name} has invalid dimensions")
    if any(isinstance(x, bool) or not isinstance(x, Real) for x in values):
        raise ValueError(f"{name} must be numeric")
    try:
        values = tuple(float(x) for x in values)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{name} must be finite")
    return values


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


@dataclass(frozen=True)
class ExecutionFeedback:
    episode_id: str
    request_id: str
    requested_action: tuple[float, ...]
    applied_action: tuple[float, ...] | None
    requested_at: float
    effective_at: float | None
    status: str
    source: str = "simulated"
    previous_action: tuple[float, ...] | None = None
    duration_seconds: float | None = None

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (self.episode_id, self.request_id)):
            raise ValueError("episode_id and request_id must be nonempty")
        if self.source not in ("simulated", "rf_emulator", "field"):
            raise ValueError("source must explicitly identify the evidence type")
        if self.status not in ("applied", "rejected", "timeout"):
            raise ValueError("unknown execution status")
        requested = _action(self.requested_action, "requested_action")
        object.__setattr__(self, "requested_action", requested)
        _time(self.requested_at, "requested_at")
        if self.previous_action is not None:
            object.__setattr__(
                self,
                "previous_action",
                _action(self.previous_action, "previous_action", len(requested)),
            )
        if self.status == "applied":
            if (
                self.applied_action is None
                or self.effective_at is None
                or self.duration_seconds is None
            ):
                raise ValueError("applied status requires actual applied action and effective time")
            object.__setattr__(
                self,
                "applied_action",
                _action(self.applied_action, "applied_action", len(requested)),
            )
            _time(self.effective_at, "effective_at")
            _time(self.duration_seconds, "duration_seconds")
            if self.duration_seconds == 0:
                raise ValueError("applied duration must be positive")
            if self.effective_at < self.requested_at:
                raise ValueError("effective time cannot precede request")
        elif any(
            x is not None for x in (self.applied_action, self.effective_at, self.duration_seconds)
        ):
            raise ValueError("unconfirmed requests cannot carry applied-action labels")


@dataclass(frozen=True)
class TransitionRecord:
    feedback: ExecutionFeedback
    state: tuple[float, ...]
    next_state: tuple[float, ...]
    observed_at: float
    available_at: float
    next_observed_at: float
    next_available_at: float
    reward: float
    group_id: str

    def __post_init__(self):
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError(
                "group_id must identify related scenarios/branch origins for splitting"
            )
        if not isinstance(self.feedback, ExecutionFeedback) or self.feedback.status != "applied":
            raise ValueError("training transitions require confirmed applied actions")
        for name in ("state", "next_state"):
            values = tuple(getattr(self, name))
            if not values or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in values
            ):
                raise ValueError(f"{name} must contain finite numeric observations")
            object.__setattr__(self, name, values)
        if len(self.state) != len(self.next_state):
            raise ValueError("state and next_state dimensions must match")
        for name in ("observed_at", "available_at", "next_observed_at", "next_available_at"):
            _time(getattr(self, name), name)
        if not (
            self.observed_at
            <= self.available_at
            <= self.feedback.requested_at
            <= self.feedback.effective_at
            < self.next_observed_at
            <= self.next_available_at
        ):
            raise ValueError("causal available/effective/observation times are inconsistent")
        if self.next_observed_at > self.feedback.effective_at + self.feedback.duration_seconds:
            raise ValueError("next observation is outside the confirmed action duration")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise ValueError("reward must be numeric")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")


def _validated_records(records, spec):
    records = tuple(records)
    if not records:
        raise ValueError("training record collection cannot be empty")
    seen = set()
    for record in records:
        if not isinstance(record, TransitionRecord):
            raise ValueError("expected a validated TransitionRecord")
        key = (record.feedback.episode_id, record.feedback.request_id)
        if key in seen:
            raise ValueError("duplicate episode/request identity")
        seen.add(key)
        spec.validate(record.feedback.applied_action, previous=record.feedback.previous_action)
    return records


def write_records(path, records, spec, *, overwrite=False):
    """Validate the whole collection before atomically publishing a JSONL file."""
    records = _validated_records(records, spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            {"schema_version": 1, "action_spec": spec.name, **asdict(record)},
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
        for record in records
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Linking a complete temporary prevents races from overwriting existing data.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_transitions(path, spec):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                version = row.pop("schema_version")
                if type(version) is not int or version != 1 or row.pop("action_spec") != spec.name:
                    raise ValueError("incompatible schema or action specification")
                row["feedback"] = ExecutionFeedback(**row["feedback"])
                records.append(TransitionRecord(**row))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid transition at line {number}: {error}") from error
    return _validated_records(records, spec)
