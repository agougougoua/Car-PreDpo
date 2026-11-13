# Copyright 2025 the LlamaFactory team.
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
import torch.distributed as dist
_orig_all_gather = dist.all_gather_into_tensor

def _patched_all_gather(output_tensor, input_tensor, *, group=None, async_op=False):
    if output_tensor.dtype != input_tensor.dtype:
        input_tensor = input_tensor.to(output_tensor.dtype)
    return _orig_all_gather(output_tensor, input_tensor, group=group, async_op=async_op)

dist.all_gather_into_tensor = _patched_all_gather

from llamafactory.train.tuner import run_exp
#上面代码适应dz_z3,在 all_gather 之前把 input 强制 cast 成 output 的 dtype，绕过 PyTorch 的严格检查

def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
