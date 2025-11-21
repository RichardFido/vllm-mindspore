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
'''
common utils
'''

import contextlib
import logging
import os
import yaml
import signal
import psutil
import subprocess
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = {}

HAS_MODEL_PATH_REGISTERED = False


def register_model_path_from_yaml(yaml_file):
    """
    Register the model path to MODEL_PATH dict.
    """
    global HAS_MODEL_PATH_REGISTERED
    if not HAS_MODEL_PATH_REGISTERED:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        module_info_yaml_path = os.path.join(current_dir, yaml_file)
        with open(module_info_yaml_path) as f:
            models = yaml.safe_load(f)

        MODEL_PATH.update({
            model_name:
            f"/home/workspace/mindspore_dataset/weight/{model_name}"
            for model_name in models
        })
        HAS_MODEL_PATH_REGISTERED = True


register_model_path_from_yaml("model_info.yaml")


def setup_function():
    """pytest will call the setup_function before case executes."""
    device_id = os.environ.pop("DEVICE_ID", None)
    if device_id is not None:
        # Specify device through environment variables to avoid the problem
        # of delayed resource release in single card cases. When executing
        # this function, DEVICE_ID has already taken effect, and the device id
        # needs to be reset to 0, otherwise it may be out of bounds.
        import mindspore as ms
        ms.set_device("Ascend", 0)
        logger.warning("This case is assigned to device:%s", str(device_id))
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_id

    # Randomly specify LCCL and HCCL ports for cases without specified port,
    # mainly in single card concurrent scenarios, to avoid port conflicts.
    lccl_port = os.getenv("LCAL_COMM_ID", None)
    if not lccl_port:
        lccl_port = random.randint(61000, 65535)
        os.environ["LCAL_COMM_ID"] = f"127.0.0.1:{lccl_port}"

    hccl_port = os.getenv("HCCL_IF_BASE_PORT", None)
    if not hccl_port:
        hccl_port = random.randint(61000, 65535)
        os.environ["HCCL_IF_BASE_PORT"] = str(hccl_port)


def cleanup_subprocesses(pid=None) -> None:
    """Cleanup all subprocesses raise by main test process."""
    pid = pid if pid else os.getpid()
    cur_proc = psutil.Process(pid)
    children = cur_proc.children(recursive=True)
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)


def teardown_function():
    """pytest will call the teardown_function after case function completed."""
    cleanup_subprocesses()
