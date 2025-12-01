# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/core/sched/scheduler.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""Enhance system availability when error occurs in model-execution."""

# noqa: G004

from collections import defaultdict

from vllm.logger import init_logger
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def update_from_output(
    self,
    scheduler_output,
    model_runner_output,
) -> dict[int, EngineCoreOutputs]:
    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    pooler_outputs = model_runner_output.pooler_output
    num_nans_in_logits = model_runner_output.num_nans_in_logits
    kv_connector_output = model_runner_output.kv_connector_output

    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    spec_decoding_stats = None
    kv_connector_stats = (kv_connector_output.kv_connector_stats
                          if kv_connector_output else None)

    # Add by vllm-mindspore begin:
    running_req_ids = [req.request_id for req in self.running]
    # abort_req_ids used to keep track of failed requests
    # caused by model execution exception
    abort_req_ids: list[str] = []
    # Add by vllm-mindspore end.

    # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
    # the below loop can be a performance bottleneck. We should do our best
    # to avoid expensive operations inside the loop.
    stopped_running_reqs: set[Request] = set()
    stopped_preempted_reqs: set[Request] = set()
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        assert num_tokens_scheduled > 0
        request = self.requests.get(req_id)
        if request is None:
            # The request is already finished. This can happen if the
            # request is aborted while the model is executing it (e.g.,
            # in pipeline parallelism).
            continue

        # Add by vllm-mindspore begin:
        # None sampled_token_ids comes from exception model execution,
        # set them to abort list
        # to keep main scheduler task running right.
        if sampled_token_ids is None:
            logger.warning(
                'Process aborted request %s from running requests %s', req_id,
                running_req_ids)
            outputs[request.client_index].append(
                EngineCoreOutput(request_id=req_id,
                                 new_token_ids=[],
                                 finish_reason=FinishReason.ABORT,
                                 stop_reason=request.stop_reason,
                                 events=request.take_events()))
            abort_req_ids.append(req_id)
            continue
        # Add by vllm-mindspore end.

        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[
            req_index] if sampled_token_ids else []

        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id))
        if scheduled_spec_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            # num_computed_tokens represents the number of tokens
            # processed in the current step, considering scheduled
            # tokens and rejections. If some tokens are rejected,
            # num_computed_tokens is decreased by the number of rejected
            # tokens.
            request.num_computed_tokens -= num_rejected
            spec_decoding_stats = self.make_spec_decoding_stats(
                spec_decoding_stats,
                num_draft_tokens=num_draft_tokens,
                num_accepted_tokens=num_accepted)

        stopped = False
        new_logprobs = None
        new_token_ids = generated_token_ids
        kv_transfer_params = None
        status_before_stop = request.status

        # Check for stop and update request status.
        if new_token_ids:
            new_token_ids, stopped = self._update_request_with_output(
                request, new_token_ids)

        # Stop checking for pooler models.
        pooler_output = None
        if pooler_outputs:
            pooler_output = pooler_outputs[req_index]
            stopped = check_stop(request, self.max_model_len, pooler_output)

        if stopped:
            kv_transfer_params = self._free_request(request)
            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

        # Extract sample logprobs if needed.
        if request.sampling_params is not None \
            and request.sampling_params.logprobs is not None and logprobs:
            # NOTE: once we support N tokens per step (spec decode),
            # the outer lists can be of length > 1.
            new_logprobs = logprobs.slice(req_index, req_index + 1)

        if new_token_ids and self.structured_output_manager.should_advance(
                request):
            # should not be None if use_structured_output, we have
            # checked above, so safe to ignore type warning
            request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                req_id, new_token_ids)

        if num_nans_in_logits is not None and req_id in num_nans_in_logits:
            request.num_nans_in_logits = num_nans_in_logits[req_id]

        # Get prompt logprobs for this request.
        prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
        if new_token_ids or pooler_output is not None \
            or kv_transfer_params:

            # Add EngineCoreOutput for this Request.
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=request.get_finished_reason(),
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    kv_transfer_params=kv_transfer_params,
                    trace_headers=request.trace_headers,
                    num_cached_tokens=request.num_cached_tokens,
                ))
        else:
            # Invariant: EngineCore returns no partial prefill outputs.
            assert not prompt_logprobs_tensors

    # Add by vllm-mindspore begin:
    # make failed requests finished to make the server
    # can continue to process new request
    if len(abort_req_ids) > 0:
        logger.warning('Aborted requests are %s', abort_req_ids)
        self.finish_requests(abort_req_ids, RequestStatus.FINISHED_ABORTED)
    # Add by vllm-mindspore end.

    # Remove the stopped requests from the running and waiting queues.
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)
    if stopped_preempted_reqs:
        # This is a rare case and unlikely to impact performance.
        self.waiting.remove_requests(stopped_preempted_reqs)

    # KV Connector: update state for finished KV Transfers.
    if model_runner_output.kv_connector_output:
        self._update_from_kv_xfer_finished(
            model_runner_output.kv_connector_output)

    # Create EngineCoreOutputs for all clients that have requests with
    # outputs in this step.
    engine_core_outputs = {
        client_index: EngineCoreOutputs(outputs=outs)
        for client_index, outs in outputs.items()
    }

    finished_req_ids = self.finished_req_ids_dict
    if finished_req_ids:
        # Include ids of requests that finished since last outputs
        # were sent.
        for client_index, finished_set in finished_req_ids.items():
            # Set finished request set in EngineCoreOutputs for this client.
            if (eco := engine_core_outputs.get(client_index)) is not None:
                eco.finished_requests = finished_set
            else:
                engine_core_outputs[client_index] = EngineCoreOutputs(
                    finished_requests=finished_set)
        finished_req_ids.clear()

    if (stats := self.make_stats(spec_decoding_stats,
                                 kv_connector_stats)) is not None:
        # Return stats to only one of the front-ends.
        if (eco := next(iter(engine_core_outputs.values()), None)) is None:
            # We must return the stats even if there are no request
            # outputs this step.
            engine_core_outputs[0] = eco = EngineCoreOutputs()
        eco.scheduler_stats = stats

    return engine_core_outputs
