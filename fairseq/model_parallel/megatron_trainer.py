# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

from fairseq import distributed_utils
from fairseq.trainer import Trainer

try:
    from fairseq.model_parallel.megatron import mpu
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class MegatronTrainer(Trainer):
    """Main class for model parallel with data parallel training."""
    def __init__(self, args, task, model, criterion):
        if not has_megatron_submodule:
            raise ImportError(
                '\n\nPlease install the megatron submodule:'
                '\n\n  git submodule update --init '
                'fairseq/model_parallel/megatron'
            )
        super().__init__(args, task, model, criterion)

    def clip_grad_norm(self, clip_norm):
        def _aggregate_model_parallel_grad_norm(total_norm):
            total_norm = total_norm ** 2
            distributed_utils.all_reduce(
                total_norm, group=distributed_utils.get_model_parallel_group()
            )
            total_norm = total_norm ** 0.5
            return total_norm
        return self.optimizer.clip_grad_norm(
            clip_norm,
            aggregate_norm_fn=_aggregate_model_parallel_grad_norm,
        )
