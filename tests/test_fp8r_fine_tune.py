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
"""FP8r for fine-tunes of the released architecture: their own action stream and width, quantized from BF16, with
real-time-chunking prefixes modulated clean per token. Built on the meta device, so no weights or GPU are needed."""

from __future__ import annotations

import pytest
import torch

from flux_action.config import REQUIRED_CONTENT_STREAMS
from flux_action.models.transformer import JointSingleSeqParams
from flux_action.models.transformer_inf_fp8r import HIDDEN, FP8RInferenceDiT, _PreparedRequest
from flux_action.models.wiring import action_dit_params, build_action_dit, restrict_content_streams

STREAM, WIDTH, CHUNK = "action_fine_tune", 20, 32


@pytest.fixture(scope="module")
def fine_tune():
    params = action_dit_params(
        restrict_content_streams(JointSingleSeqParams(), REQUIRED_CONTENT_STREAMS), STREAM, WIDTH
    )
    reference = build_action_dit(params, modality=STREAM, device="meta")
    return FP8RInferenceDiT.from_bf16_state_dict(reference.state_dict(), action=STREAM)


def test_a_fine_tune_quantizes_with_its_own_action_stream(fine_tune):
    assert fine_tune.emb_in[STREAM].in_features == fine_tune.emb_in[f"{STREAM}_cond"].in_features == WIDTH
    assert fine_tune.final_layer[STREAM].linear.out_features == WIDTH
    assert set(fine_tune.final_layer) == {"video", STREAM}
    # The action stream's boundaries stay BF16; the trunk is E4M3.
    assert fine_tune.emb_in[STREAM].weight.dtype == torch.bfloat16
    assert fine_tune.single_blocks[0].projections.qkv_q.dtype == torch.float8_e4m3fn


def test_clean_action_tokens_take_timestep_zeros_modulations(fine_tune, monkeypatch):
    def step_math(vector, video_timestep, action_timestep):
        # Every value carries the timestep it was computed for, as (batch, 1, hidden).
        def value(t):
            return t[:, None, None].expand(1, 1, HIDDEN).clone()

        return tuple(
            value(video_timestep if index in (0, 1, 2, 6, 7, 8, 12, 13) else action_timestep)
            for index in range(16)
        )

    monkeypatch.setattr(fine_tune, "_compiled_step", step_math)
    request = _PreparedRequest(
        1, torch.bfloat16, torch.device("cpu"), (8, 4, 4, CHUNK, 1), *([None] * 4), (), torch.zeros(1, HIDDEN)
    )
    times = torch.tensor([0.9, 0.5])
    plain = fine_tune.prepare_steps(request, times, times)
    pinned = fine_tune.prepare_steps(request, times, times, clean_action_tokens=5)
    for t, step, prefixed in zip(times, plain, pinned, strict=True):
        assert step.action_early[0].shape == (1, 1, HIDDEN)
        for value in (*prefixed.action_early, *prefixed.dynamic_joint[1], *prefixed.final_action):
            assert value.shape == (1, CHUNK, HIDDEN)
            assert (value[:, :5] == 0).all() and (value[:, 5:] == t).all()
        for value in (*prefixed.video_early, *prefixed.dynamic_joint[0], *prefixed.final_video):
            assert value.shape == (1, 1, HIDDEN) and (value == t).all()
