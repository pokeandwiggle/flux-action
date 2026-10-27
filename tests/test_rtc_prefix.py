# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train-time real-time chunking: clean action prefixes with per-token timesteps, out of the loss, pinned at
inference."""

from __future__ import annotations

import pytest
import torch
from conftest import make_policy, tiny_config

from flux_action.processing import packing

CHUNK, D = 32, 6


class DitSpy:
    """Wraps the policy's DiT and records every call's keyword arguments."""

    def __init__(self, policy):
        self.calls = []
        self.dit = policy.dit
        original = policy.dit.forward

        def forward(*args, **kwargs):
            self.calls.append(
                {k: v.detach().clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            )
            return original(*args, **kwargs)

        policy.dit.forward = forward


def test_prefix_timesteps_pin_each_samples_first_tokens():
    t = torch.tensor([0.5, 0.75, 0.25])
    delays = torch.tensor([0, 2, 4])
    timesteps = packing.prefix_timesteps(t, delays, 5)
    assert timesteps.tolist() == [
        [0.5] * 5,
        [0.0, 0.0, 0.75, 0.75, 0.75],
        [0.0, 0.0, 0.0, 0.0, 0.25],
    ]
    assert packing.prefix_loss_mask(delays, 5).tolist() == [
        [1.0] * 5,
        [0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]


def test_per_token_timesteps_keep_the_prefix_clean():
    x0 = torch.randn(2, 4, 3)
    t = packing.prefix_timesteps(torch.tensor([0.7, 0.7]), torch.tensor([1, 3]), 4)
    noised, _ = packing.add_noise(x0, t)
    assert torch.equal(noised[0, :1], x0[0, :1]) and torch.equal(noised[1, :3], x0[1, :3])
    assert not torch.equal(noised[0, 1:], x0[0, 1:])


@pytest.mark.parametrize("reduction", ["joint_tokens", "modalities"])
def test_prefix_tokens_carry_no_loss(reduction):
    target = {"x_video": torch.zeros(1, 3, 2), "x_action": torch.zeros(1, 8, D)}
    pred = {"x_video": torch.zeros(1, 3, 2), "x_action": torch.zeros(1, 8, D)}
    pred["x_action"][0, :2] = 1.0  # window 1's prefix
    pred["x_action"][0, 4:7] = 1.0  # window 2's prefix
    mask = packing.prefix_loss_mask(torch.tensor([2, 3]), 4)
    losses = packing.flow_loss(pred, target, "action", action_token_mask=mask, reduction=reduction)
    assert losses["loss"].item() == 0.0 and losses["action_mse"].item() == 0.0
    pred["x_action"][0, 3] = 1.0  # a denoised token of window 1
    losses = packing.flow_loss(pred, target, "action", action_token_mask=mask, reduction=reduction)
    assert losses["action_mse"].item() == pytest.approx(1.0 / 3.0)  # 3 trained tokens, one wrong


def test_an_all_ones_token_mask_is_the_plain_loss():
    pred = {"x_video": torch.randn(1, 3, 2), "x_action": torch.randn(1, 8, D)}
    target = {k: torch.randn_like(v) for k, v in pred.items()}
    plain = packing.flow_loss(pred, target, "action")
    masked = packing.flow_loss(pred, target, "action", action_token_mask=torch.ones(2, 4))
    assert masked["loss"].item() == pytest.approx(plain["loss"].item())


def test_training_pins_a_clean_prefix_at_timestep_zero(batch):
    torch.manual_seed(3)
    policy = make_policy(tiny_config(rtc_max_delay=8)).train()
    spy = DitSpy(policy)
    batch = {
        k: (v.repeat(4, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor) else v * 4)
        for k, v in batch.items()
    }
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    (call,) = spy.calls
    timesteps = call["x_action_timesteps"].reshape(4, CHUNK)
    delays = (timesteps == 0).sum(-1)
    assert (delays <= 8).all() and delays.max() > 0
    for row, d in zip(timesteps, delays.tolist(), strict=True):
        assert (row[:d] == 0).all() and (row[d:] > 0).all()


def test_evaluation_trains_no_prefix(batch):
    policy = make_policy(tiny_config(rtc_max_delay=8)).eval()
    spy = DitSpy(policy)
    policy(batch)
    assert (spy.calls[0]["x_action_timesteps"] > 0).all()


def test_inference_pins_the_prefix_and_returns_it_verbatim(batch):
    policy = make_policy(tiny_config(rtc_max_delay=8))
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    prefix = torch.rand(1, 5, D)
    spy = DitSpy(policy)
    chunk = policy.predict_action_chunk(observation, action_prefix=prefix)
    assert chunk.shape == (1, CHUNK, D) and torch.equal(chunk[:, :5], prefix)
    assert len(spy.calls) == policy.config.num_inference_steps
    for call in spy.calls:
        assert (call["x_action_timesteps"][0, :5] == 0).all()
        assert (call["x_action_timesteps"][0, 5:] > 0).all()
        torch.testing.assert_close(call["x_action"][0, :5], prefix[0] * policy.config.action_scale)
    unpinned = policy.predict_action_chunk(observation)
    assert not torch.equal(unpinned[:, :5], prefix)


def test_a_prefix_longer_than_trained_is_refused(batch):
    policy = make_policy(tiny_config(rtc_max_delay=4))
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    with pytest.raises(ValueError, match="rtc_max_delay 4"):
        policy.predict_action_chunk(observation, action_prefix=torch.rand(1, 5, D))


@pytest.mark.parametrize("delay", [-1, CHUNK])
def test_rtc_max_delay_must_leave_actions_to_predict(delay):
    with pytest.raises(ValueError, match="rtc_max_delay"):
        tiny_config(rtc_max_delay=delay)


def test_delay_probs_draw_only_the_delays_they_weight():
    delays = packing.sample_rtc_delays(4000, 8, [0.75] + [0.0] * 7 + [0.25], torch.Generator().manual_seed(0))
    assert set(delays.tolist()) == {0, 8}
    assert (delays == 8).float().mean().item() == pytest.approx(0.25, abs=0.03)


def test_training_draws_delays_from_the_configured_probs(batch):
    torch.manual_seed(3)
    policy = make_policy(tiny_config(rtc_max_delay=8, rtc_delay_probs=[0.0] * 8 + [1.0])).train()
    spy = DitSpy(policy)
    batch = {
        k: (v.repeat(4, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor) else v * 4)
        for k, v in batch.items()
    }
    policy(batch)
    timesteps = spy.calls[0]["x_action_timesteps"].reshape(4, CHUNK)
    assert ((timesteps == 0).sum(-1) == 8).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"rtc_max_delay": 0, "rtc_delay_probs": [1.0]},
        {"rtc_max_delay": 4, "rtc_delay_probs": [0.75, 0.25]},
        {"rtc_max_delay": 1, "rtc_delay_probs": [0.5, 0.25]},
        {"rtc_max_delay": 1, "rtc_delay_probs": [1.5, -0.5]},
    ],
)
def test_rtc_delay_probs_must_weight_every_trained_delay(overrides):
    with pytest.raises(ValueError, match="rtc_delay_probs"):
        tiny_config(**overrides)
