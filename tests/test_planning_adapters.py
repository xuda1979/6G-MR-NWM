import pytest
import torch

from mr_nwm.model import ModelConfig, MultiRateNestedWorldModel
from mr_nwm.planning_adapters import WorldModelPredictor, power_action_spec


def test_real_reference_rollout_adapter_pads_horizon_and_restores_module_modes():
    model = MultiRateNestedWorldModel(ModelConfig(horizon=2))
    model.train()
    model.backbone.eval()
    before = {name: module.training for name, module in model.named_modules()}
    predictor = WorldModelPredictor(model, torch.tensor([[1, 2]]), torch.zeros(1, 4))
    state = (0.5, 0.5, 0.4, 0.4, 0.1, 0.1)
    actions = ((0.1, 0.4), (0.4, 0.1), (0.1, 0.1))
    after = predictor(state, actions)
    assert len(after) == 3 and all(len(row) == 6 for row in after)
    assert all(torch.isfinite(torch.tensor(row)).all() for row in after)
    assert before == {name: module.training for name, module in model.named_modules()}
    assert all(parameter.grad is None for parameter in model.parameters())
    again = predictor(state, actions)
    assert again == after


def test_reference_adapter_validates_finite_and_dimensioned_inputs():
    model = MultiRateNestedWorldModel(ModelConfig(horizon=2))
    predictor = WorldModelPredictor(model, torch.tensor([[1]]), torch.zeros(1, 4))
    with pytest.raises(ValueError):
        predictor((0,) * 6, ())
    with pytest.raises(ValueError):
        predictor((0,) * 6, ((float("nan"), 0),))
    with pytest.raises(ValueError):
        predictor((0,) * 5, ((0.1, 0.1),))


def test_slew_and_power_budget_are_explicit_capability_constraints():
    spec = power_action_spec(links=2, total_power=0.8, max_delta=(0.2, 0.2))
    spec.validate((0.2, 0.4), previous=(0.1, 0.3))
    with pytest.raises(ValueError):
        spec.validate((0.7, 0.1), previous=(0.1, 0.3))
    with pytest.raises(ValueError):
        spec.validate((0.7, 0.4), previous=(0.6, 0.4))
