# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/block_table.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

import mindspore as ms
import numpy as np
from vllm.distributed import get_dcp_group


def BlockTable__init__(
    self,
    block_size: int,
    max_num_reqs: int,
    max_num_blocks_per_req: int,
    max_num_batched_tokens: int,
    pin_memory: bool,
    device,
):
    self.block_size = block_size
    self.max_num_reqs = max_num_reqs
    self.max_num_blocks_per_req = max_num_blocks_per_req
    self.max_num_batched_tokens = max_num_batched_tokens
    self.pin_memory = pin_memory
    self.device = device
    self.block_table = self._make_buffer(max_num_reqs,
                                         max_num_blocks_per_req,
                                         dtype=ms.int32)
    self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
    self.slot_mapping = self._make_buffer(
        self.max_num_batched_tokens,
        # vllm-mindspore begin
        # for reshapeandcache's input, slot_mapping should be int32
        dtype=ms.int32)
    # vllm-mindspore end
    try:
        self.dcp_world_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group
    except AssertionError:
        # DCP might not be initialized in testing
        self.dcp_world_size = 1
        self.dcp_rank = 0
