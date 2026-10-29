"""Finite-search adapters for existing MR-NWM models; no radio execution API."""

import copy
import math

import torch

from .action_space import ActionSpec


def power_action_spec(*, links=2, total_power=0.8, max_delta=None):
    if isinstance(links, bool) or not isinstance(links, int) or not 1 <= links <= 64:
        raise ValueError("links must be an integer in [1, 64]")
    return ActionSpec(
        name="normalized_power_v1",
        names=tuple(f"power_{i}" for i in range(links)),
        units=("normalized_power",) * links,
        lower=(0.0,) * links,
        upper=(1.0,) * links,
        sum_indices=tuple(range(links)),
        sum_limit=total_power,
        max_delta=max_delta,
    )


class WorldModelPredictor:
    """Isolated, inference-only batch-one rollout callable for BoundedPlanner.

    Supports the reference and language-foundation rollout signatures. Multimodal
    multi-window forecasts keep the foundation API's explicit physical clock rule.
    Do not share one mutable model instance between concurrent planning threads.
    """

    def __init__(self, model, intents, context, *, macro_period_seconds=None):
        self.model = model
        self.intents = copy.deepcopy(intents)
        parameter = next(model.parameters())
        self.context = torch.as_tensor(
            context, device=parameter.device, dtype=parameter.dtype
        ).clone()
        if self.context.shape != (1, model.config.context_dim):
            raise ValueError("context must have shape [1, context_dim]")
        if not torch.isfinite(self.context).all():
            raise ValueError("context must be finite")
        if macro_period_seconds is not None and (
            not math.isfinite(macro_period_seconds) or macro_period_seconds <= 0
        ):
            raise ValueError("macro_period_seconds must be positive and finite")
        self.macro_period_seconds = macro_period_seconds

    def __call__(self, initial_state, actions):
        c = self.model.config
        if not actions:
            raise ValueError("actions cannot be empty")
        parameter = next(self.model.parameters())
        state = torch.as_tensor(initial_state, device=parameter.device, dtype=parameter.dtype)
        action_tensor = torch.as_tensor(actions, device=parameter.device, dtype=parameter.dtype)
        if state.shape != (c.state_dim,) or action_tensor.shape != (len(actions), c.action_dim):
            raise ValueError("state/action dimensions do not match model")
        if not torch.isfinite(state).all() or not torch.isfinite(action_tensor).all():
            raise ValueError("state/actions must be finite")
        steps = len(actions)
        padding = (-steps) % c.horizon
        if padding:
            action_tensor = torch.cat([action_tensor, action_tensor[-1:].repeat(padding, 1)])
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            kwargs = (
                {"macro_period_seconds": self.macro_period_seconds}
                if self.macro_period_seconds is not None
                else {}
            )
            with torch.inference_mode():
                predicted = self.model.rollout(
                    copy.deepcopy(self.intents),
                    state[None].clone(),
                    action_tensor[None].clone(),
                    self.context.clone(),
                    **kwargs,
                )
            if predicted.shape != (1, steps + padding, c.state_dim):
                raise ValueError("world model returned malformed rollout")
            if not torch.isfinite(predicted).all():
                raise ValueError("world model returned nonfinite predictions")
            return tuple(tuple(row) for row in predicted[0, :steps].cpu().tolist())
        finally:
            for module, training in modes:
                module.training = training
