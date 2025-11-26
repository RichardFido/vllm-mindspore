# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
import pytest
import os
import sys
import json
import requests
from unittest.mock import patch

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH, start_vllm_server,
                                      get_key_counter_from_log,
                                      stop_vllm_server, send_and_get_request)

import vllm_mindspore
from vllm import LLM, SamplingParams
from openai import OpenAI

# def env
env_vars = {
    "VLLM_MS_MODEL_BACKEND": "Native",
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}

QWEN_7B_MODEL = MODEL_PATH["Qwen2.5-7B-Instruct"]
QWEN_32B_MODEL = MODEL_PATH["Qwen2.5-32B-Instruct"]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_001():
    """
    Test Summary:
        离线ms场景使用最简配置，prompts传入字符串，
        sampling_params为None
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    prompts = "Today is"
    sampling_params = None
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_002():
    """
    Test Summary:
        离线ms场景sampling_params配置支持的4个后处理参数
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    prompts = "Today is"
    sampling_params = SamplingParams(n=3,
                                     top_k=3,
                                     top_p=0.5,
                                     temperature=2.0,
                                     repetition_penalty=2.0)
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        assert prompt == prompts
        assert output.finished
        assert len(output.outputs) == 3


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_003():
    """
    Test Summary:
        离线ms场景llm.generate的promtps是空字符串
    Expected Result:
        运行报错 ValueError
    Model Info:
        Qwen2.5-7B-Instruct
    """
    prompts = ""
    sampling_params = SamplingParams(top_k=1)
    llm = LLM(model=QWEN_7B_MODEL)
    with pytest.raises(ValueError) as err:
        llm.generate(prompts, sampling_params)
    assert "The decoder prompt cannot be empty" in str(err.value)
    prompts = ["", "Today is", "Llama is"]
    with pytest.raises(ValueError) as err:
        llm.generate(prompts, sampling_params)
    assert "The decoder prompt cannot be empty" in str(err.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_004():
    """
    Test Summary:
        离线ms场景llm.generate的promtps 是list(str)
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    prompts = ["I am", "Today is", "I love Beijing, because"]
    sampling_params = SamplingParams(temperature=0.0, logprobs=4)
    llm = LLM(model=QWEN_7B_MODEL)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        assert output.finished
        for i in range(len(output.outputs[0].token_ids)):
            assert len(output.outputs[0].logprobs[i]) >= 4
    assert outputs[2].outputs[0].text == \
           " it is a city with a long history. " + \
           "Which of the following options correctly expresses"


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_001():
    """
    Test Summary:
        ms服务化+请求接口,使用最简配置，prompts传入字符串
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    openai_api_key = "EMPTY"
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        openai_api_base = f'http://localhost:{serve_port}/v1'
    else:
        openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    _ = client.completions.create(model=model, prompt="Today is")
    stop_vllm_server(process)
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_002():
    """
    Test Summary:
        ms服务化+请求接口,测试prompts为字符串列表
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_002.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["I am", "Today is", "Llama is"],
        "top_k": 1
    }
    json_data = json.dumps(data)
    response = requests.post(url,
                             data=json_data,
                             headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.allcards
@patch.dict(os.environ, env_vars)
def test_vllm_server_003():
    """
    Test Summary:
        ms服务化+请求接口,后处理参数组合测试,测试repetition_penalty
        temperature、top_k、top_p使用默认值
        repetition_penalty配置为0.5 ，1.5，  2，  2.5
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-32B-Instruct
    """
    log_name = "test_vllm_server_003.log"
    model = QWEN_32B_MODEL
    extra_params = "--tensor_parallel_size=8"
    process = start_vllm_server(model, log_name, extra_params=extra_params)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": ["I am", "Today is", "Llama is"],
        "repetition_penalty": 0.5
    }
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 1.5
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 2
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["repetition_penalty"] = 2.5
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I love Beijing, because",
        "temperature": 0
    }
    json_data = json.dumps(data)
    response5 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200
    assert response5.json()["choices"][0]["text"] == \
           " it is the capital of China. Which " \
           "preposition should be used to fill in"
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_004():
    """
    Test Summary:
        ms服务化+请求接口,后处理参数组合测试,测试temperature
        repetition_penalty、top_k、top_p使用默认值
        temperature配置为0， 0.0001，  0.001， 2
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_004.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": [15364, 374], "temperature": 0}
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 0.0001
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 0.001
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["temperature"] = 2
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_005():
    """
    Test Summary:
        ms服务化+请求接口,后处理参数组合测试,测试top_k
        repetition_penalty、temperature、top_p使用默认值
        topk覆盖0，最大int，词表长度， 词表长度-1
    Expected Result:
        除长度超出词表大小外,其余场景运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_005.log"
    model = QWEN_7B_MODEL
    with open(os.path.join(model, "tokenizer.json")) as f:
        tokens = json.load(f)
    vocab_len = len(tokens['model']['vocab'])
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": [[15364, 374], [15364, 374]], "top_k": 0}

    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    # Far larger than the size of the vocab_len
    data["top_k"] = sys.maxsize
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_k"] = vocab_len
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_k"] = vocab_len - 1
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 400
    assert response3.status_code == 200
    assert response4.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_006():
    """
    Test Summary:
        ms服务化+请求接口,后处理参数组合测试,测试top_p
        repetition_penalty、temperature、top_k使用默认值
        top_p覆盖0， 0.3， 0.5， 0.8
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_006.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {"model": model, "prompt": "I am", "top_p": 0}
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.3
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.5
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data["top_p"] = 0.8
    json_data = json.dumps(data)
    response4 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})

    stop_vllm_server(process)
    assert response1.status_code == 400
    assert response2.status_code == 200
    assert response3.status_code == 200
    assert response4.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_server_007():
    """
    Test Summary:
        ms服务化+请求接口,后处理参数组合测试,其他组合场景
        repetition_penalty=1.5 temperature=0.001 top_k=5 top_p=0.5
        repetition_penalty=1 temperature=2 top_k=vocabSize-1 top_p=1
        repetition_penalty=2 temperature=0.001 top_k=1 top_p=1
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_server_007.log"
    model = QWEN_7B_MODEL
    with open(os.path.join(model, "tokenizer.json")) as f:
        tokens = json.load(f)
    vocab_len = len(tokens['model']['vocab'])
    process = start_vllm_server(model, log_name)
    serve_port = os.getenv("TEST_SERVE_PORT", None)
    if serve_port:
        url = f'http://localhost:{serve_port}/v1/completions'
    else:
        url = 'http://localhost:8000/v1/completions'
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 1.5,
        "temperature": 0.001,
        "top_k": 5,
        "top_p": 0.5
    }
    json_data = json.dumps(data)
    response1 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 1,
        "temperature": 2,
        "top_k": vocab_len - 1,
        "top_p": 1
    }
    json_data = json.dumps(data)
    response2 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    data = {
        "model": model,
        "prompt": "I am",
        "repetition_penalty": 2,
        "temperature": 0.001,
        "top_k": 1,
        "top_p": 1
    }
    json_data = json.dumps(data)
    response3 = requests.post(url,
                              data=json_data,
                              headers={'Content-Type': 'application/json'})
    stop_vllm_server(process)
    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response3.status_code == 200
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_err_001():
    """
    Test Summary:
        ms离线场景SamplingParams的参数值异常
    Expected Result:
        报ValueError,并校验对应报错信息
    """
    with pytest.raises(ValueError) as err:
        SamplingParams(top_k=-5)
    assert "top_k must be 0 (disable), or at least 1, got -5." in str(
        err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(top_p=0)
    assert "top_p must be in (0, 1], got 0." in str(err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(temperature=-1)
    assert "temperature must be non-negative, got -1." in str(err.value)
    with pytest.raises(ValueError) as err:
        SamplingParams(repetition_penalty=-2.0)
    assert "repetition_penalty must be greater than zero, got -2.0." in str(
        err.value)
    SamplingParams(top_k=sys.maxsize)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_offline_err_002():
    """
    Test Summary:
        离线场景LLM接口的参数值异常
    Expected Result:
        报ValueError/TypeError,并校验对应报错信息
    Model Info:
        Qwen2.5-7B-Instruct
    """
    with pytest.raises(TypeError) as err:
        LLM(model=1)
    assert "expected str, bytes or os.PathLike object, not int" in str(
        err.value)
    with pytest.raises(Exception) as err:
        LLM(model="/home/workspace/mindspore_dataset/weight/")
    assert "Repo id must be in the form 'repo_name' or " \
           "'namespace/repo_name'" in str(err.value) or \
           "Invalid repository ID or local directory " \
           "specified" in str(err.value)
    llm = LLM(model=QWEN_7B_MODEL)
    with pytest.raises(TypeError) as err:
        llm.generate(1, None)
    assert "object of type 'int' has no len()" in str(err.value)
    with pytest.raises(AttributeError) as err:
        llm.generate("i am", sampling_params=1)
    assert "'int' object has no attribute 'truncate_prompt_tokens'" in str(
        err.value)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_vllm_model_alias_ms_server_001():
    """
    Test Summary:
        Natve模型运行vllm服务化场景，
        模型路径配置为/home/path,served-model-name
        配置 name1 name2 name2(name2有重复）...name8 ....(覆盖个数8个）
        发送请求使用模型名称name2
        发送请求使用模型名称name8
    Expected Result:
        运行成功，推理结果正常
    Model Info:
        Qwen2.5-7B-Instruct
    """
    log_name = "test_vllm_v073_ms_server_001.log"
    model = QWEN_7B_MODEL
    process = start_vllm_server(
        model,
        log_name,
        extra_params="--served-model-name qwen1 qwen2 qwen2 qwen3 qwen4 "
        "qwen5 qwen6 qwen7 qwen8")
    data = {
        "model": model,
        "prompt": "I am",
        "max_tokens": 100,
        "temperature": 0
    }
    response = send_and_get_request(data)
    data["model"] = "qwen2"
    response1 = send_and_get_request(data)
    data["model"] = "qwen8"
    response2 = send_and_get_request(data)
    stop_vllm_server(process)
    assert response.status_code == 404, response.text
    assert response1.status_code == 200, response1.text
    assert response2.status_code == 200, response2.text
    assert response1.json()["choices"][0]["text"] == \
           response2.json()["choices"][0]["text"]
    result = get_key_counter_from_log(log_name,
                                      "Run with native model backend")
    assert result >= 1
