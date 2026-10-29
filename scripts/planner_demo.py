"""Run bounded planning and export explicitly synthetic applied-action data.

This evaluates an exact toy simulator predictor, not a learned slice world
model. The separate power probe exercises a committed reference checkpoint.
"""

import argparse
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path

import torch

from mr_nwm.control_planner import BoundedPlanner, PlannerConfig
from mr_nwm.control_records import (
    ExecutionFeedback,
    TransitionRecord,
    read_transitions,
    write_records,
)
from mr_nwm.data import objective_features
from mr_nwm.intent import IntentTokenizer
from mr_nwm.model import ModelConfig, MultiRateNestedWorldModel
from mr_nwm.planning_adapters import WorldModelPredictor, power_action_spec
from mr_nwm.provenance import sha256_file
from mr_nwm.slicing import SliceScenario, SliceState, TwoSliceQueueEnvironment, slice_utility

ROOT = Path(__file__).resolve().parents[1]


def _episode(env, initial, *, steps, seed, name, horizon, max_sequences, exploration=None):
    planner = BoundedPlanner(env.spec, PlannerConfig(horizon=horizon, max_sequences=max_sequences))
    state, previous, records, decisions = initial, None, [], []
    for index in range(steps):
        if exploration is None:
            plan = planner.plan(
                state, env.candidates(), env.rollout, slice_utility, previous_action=previous
            )
            if plan.first_action is None:
                raise RuntimeError(f"{name} produced no executable plan: {plan.reason}")
            action = plan.first_action
            decisions.append(asdict(plan))
        else:
            action = exploration[index]
        after = env.step(state, action)
        start = index * env.scenario.step_seconds
        end = (index + 1) * env.scenario.step_seconds
        records.append(
            TransitionRecord(
                feedback=ExecutionFeedback(
                    episode_id=f"seed-{seed}/{name}",
                    request_id=f"{name}-{index}",
                    requested_action=action,
                    applied_action=action,
                    requested_at=start,
                    effective_at=start,
                    status="applied",
                    source="simulated",
                    previous_action=previous,
                    duration_seconds=env.scenario.step_seconds,
                ),
                state=state.features(),
                next_state=after.features(),
                observed_at=start,
                available_at=start,
                next_observed_at=end,
                next_available_at=end,
                reward=slice_utility(after, action),
                group_id=f"scenario-seed-{seed}",
            )
        )
        state, previous = after, action
    return records, {
        "policy": name,
        "source": "simulated",
        "predictor": "exact_two_slice_simulator",
        "steps": steps,
        "return": sum(r.reward for r in records),
        "final_queue": sum(state.queues),
        "served_payload": sum(sum(r.next_state[-4:]) for r in records),
        "unique_applied_actions": len({r.feedback.applied_action for r in records}),
        "decisions": decisions,
    }


def reference_probe(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("variant") != "full":
        raise ValueError("reference probe requires a full reference-variant checkpoint")
    model = MultiRateNestedWorldModel(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"], strict=True)
    intents = IntentTokenizer().encode(["prioritize throughput"])
    context = torch.tensor([[0.1, 0.1, 0.15, 0.05]])
    predictor = WorldModelPredictor(model, intents, context)
    candidates = tuple((a, b) for a in (0.1, 0.4, 0.7) for b in (0.1, 0.4, 0.7))

    def reward(state, action):
        with torch.inference_mode():
            return float(objective_features(torch.tensor([state]), torch.tensor([action]))[0, 0])

    plan = BoundedPlanner(power_action_spec(), PlannerConfig(horizon=2)).plan(
        (0.5, 0.5, 0.5, 0.5, 0.1, 0.1), candidates, predictor, reward
    )
    if plan.first_action is None:
        raise RuntimeError("reference checkpoint failed planner integration")
    return {
        "purpose": "existing_reference_checkpoint_interface_smoke_only",
        "checkpoint_sha256": sha256_file(checkpoint),
        "variant": saved["variant"],
        "seed": saved["seed"],
        "plan": asdict(plan),
        "field_control_evidence": False,
    }


def run_demo(output, *, seed=11, steps=8, horizon=2, max_sequences=512, overwrite=False):
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("steps must be in [1, 64]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output)
    destinations = (output / "transitions.jsonl", output / "summary.json")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("output exists; use a new directory or explicit --overwrite")
    rng = random.Random(seed)
    scenario = SliceScenario(
        total_blocks=4,
        arrivals=tuple(rng.uniform(0.2, 0.8) for _ in range(4)),
        rates=tuple(rng.uniform(0.5, 2.0) for _ in range(4)),
    )
    env = TwoSliceQueueEnvironment(scenario)
    initial = SliceState(tuple(rng.uniform(2.0, 5.0) for _ in range(4)))
    all_records, policies = [], []
    for name, length in (("greedy", 1), ("mpc", horizon)):
        records, metrics = _episode(
            env,
            initial,
            steps=steps,
            seed=seed,
            name=name,
            horizon=length,
            max_sequences=max_sequences,
        )
        all_records.extend(records)
        policies.append(metrics)
    coverage_actions = list(env.candidates())
    rng.shuffle(coverage_actions)
    records, metrics = _episode(
        env,
        initial,
        steps=len(coverage_actions),
        seed=seed,
        name="coverage",
        horizon=1,
        max_sequences=max_sequences,
        exploration=coverage_actions,
    )
    all_records.extend(records)
    policies.append(metrics)
    checkpoint = ROOT / "results/checkpoints/full_11.pt"
    probe = reference_probe(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    write_records(destinations[0], all_records, env.spec, overwrite=overwrite)
    reloaded = read_transitions(destinations[0], env.spec)
    if reloaded != tuple(all_records):
        raise RuntimeError("applied-action JSONL round trip changed records")
    source_files = (
        "src/mr_nwm/action_space.py",
        "src/mr_nwm/control_planner.py",
        "src/mr_nwm/slicing.py",
        "src/mr_nwm/control_records.py",
        "src/mr_nwm/planning_adapters.py",
        "src/mr_nwm/model.py",
        "src/mr_nwm/data.py",
        "scripts/planner_demo.py",
    )
    report = {
        "measurement_contract": "bounded-planner-synthetic-smoke-v1",
        "evidence": "simulated_only",
        "simulator_version": env.version,
        "utility": "sum_served_minus_0.5_queue_minus_0.01_allocated_blocks",
        "seed": seed,
        "planner_config": {
            "horizon": horizon,
            "max_sequences": max_sequences,
            "steps": steps,
            "discount": 1.0,
            "switch_cost": 0.0,
        },
        "scenario": asdict(scenario),
        "initial_state": asdict(initial),
        "action_spec": asdict(env.spec),
        "scheduler_codes": {"0": "round_robin", "1": "proportional_fair_score"},
        "units": "abstract payload and resource blocks; timing is uncalibrated",
        "split_group": f"scenario-seed-{seed}",
        "split_rule": "all policies and counterfactuals from this scenario stay in one split",
        "transition_rows": len(all_records),
        "transitions_sha256": sha256_file(destinations[0]),
        "policies": policies,
        "reference_probe": probe,
        "source_sha256": {path: sha256_file(ROOT / path) for path in source_files},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__},
        "limits": [
            "Exact toy simulator predicts slice consequences; no learned slice model was trained.",
            "This smoke checks machinery, not statistical gains or sim-to-real transfer.",
            "No field actions, RIC/E2 integration, deterministic deadline or NPU validation.",
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with destinations[1].open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/planner-smoke"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_demo(**vars(args))
    print(
        json.dumps(
            {
                "evidence": report["evidence"],
                "transition_rows": report["transition_rows"],
                "return": {p["policy"]: p["return"] for p in report["policies"]},
                "transitions_sha256": report["transitions_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
