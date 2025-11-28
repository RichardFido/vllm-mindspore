# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/logits_processor/builtin.py
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

from typing import Optional

import mindspore as ms
from vllm.v1.sample.logits_processor.interface import (BatchUpdate,
                                                       MoveDirectionality)


def update_state(self, batch_update: Optional[BatchUpdate]):
    if not batch_update:
        return

    needs_update = False
    # Process added requests.
    for index, params, _, _ in batch_update.added:
        min_p = params.min_p
        min_p_before = self.min_p_cpu[index]
        if min_p_before != min_p:
            needs_update = True
            self.min_p_cpu[index] = min_p
            if min_p and not min_p_before:
                self.min_p_count += 1
            elif not min_p and min_p_before:
                self.min_p_count -= 1

    if self.min_p_count:
        # Process removed requests.
        if batch_update.removed:
            needs_update = True
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_cpu[index] = 0
                    self.min_p_count -= 1

        # Process moved requests, unidirectional (a->b) and swap (a<->b).
        for adx, bdx, direct in batch_update.moved:
            min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
            if min_p_a != min_p_b:
                needs_update = True
                self.min_p_cpu[bdx] = min_p_a
                if direct == MoveDirectionality.SWAP:
                    self.min_p_cpu[adx] = min_p_b
            if direct == MoveDirectionality.UNIDIRECTIONAL:
                if min_p_a:
                    self.min_p_cpu[adx] = 0
                if min_p_b:
                    self.min_p_count -= 1

    # Update tensors if needed.
    size = batch_update.batch_size
    if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
        self.min_p = self.min_p_device[:size]
        if self.use_double_tensor:
            # vllm-mindspore: Since slicing operations cause tensors to be
            # converted to device tensors, the memory sharing mechanism of
            # NumPy becomes invalid. Therefore, values should not be accessed
            # directly from the origin tensor.
            self.min_p.copy_(ms.from_numpy(self.min_p_cpu[:size]),
                             non_blocking=True)

        self.min_p = self.min_p.unsqueeze(1)
