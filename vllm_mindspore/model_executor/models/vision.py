# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/vision.py
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from mindspore import Tensor, mint


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: Tensor,
    grid_ws: Tensor,
) -> Tensor:
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx].item() // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx].item() // spatial_merge_size
    h_index = (mint.arange(llm_grid_h).view(1, -1,
                                            1).expand(len(t_index), -1,
                                                      llm_grid_w).flatten())
    w_index = (mint.arange(llm_grid_w).view(1, 1,
                                            -1).expand(len(t_index),
                                                       llm_grid_h,
                                                       -1).flatten())
    t_index_tensor = (Tensor(t_index).view(-1, 1).expand(
        -1, llm_grid_h * llm_grid_w).long().flatten())
    _llm_pos_ids = mint.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = mint.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids
