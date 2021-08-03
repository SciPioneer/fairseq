# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn as nn

from fairseq import distributed_utils
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel
from fairseq.models import BaseFairseqModel


_GOSSIP_DISABLED = False
try:
    import gossip
except ImportError:
    _GOSSIP_DISABLED = True


def DistributedFairseqModel(args, model, process_group):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
    """
    # determine which DDP class to extend
    assert isinstance(model, nn.Module)
    if args.tpu:
        ddp_class = TPUDistributedDataParallel
        init_kwargs = dict(
            module=model,
            process_group=process_group,
        )
    elif args.distributed_wrapper == 'DDP' and args.ddp_backend == 'c10d':
        ddp_class = nn.parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
        )
        # Maintain backward compatibility
        if 'check_reduction' in inspect.getargspec(ddp_class)[0]:
            init_kwargs['check_reduction'] = True
        if 'find_unused_parameters' in inspect.getargspec(ddp_class)[0]:
            init_kwargs['find_unused_parameters'] = args.find_unused_parameters
    elif args.distributed_wrapper == 'DDP' and args.ddp_backend == 'no_c10d':
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict(
            module=model,
            buffer_size=2**28,
            process_group=process_group,
        )
    elif args.distributed_wrapper == 'SlowMo':
        if _GOSSIP_DISABLED:
            raise ImportError(
                'Cannot find gossip library. Please install from: '
                'github.com/facebookresearch/stochastic_gradient_push'
            )
        ddp_class = gossip.GossipDataParallel

        # The values of slowmo_momentum below were obtained by tuning on the
        # En-De 16 dataset by training the transformer_wmt_en_de_large model
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6

        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            nprocs_per_node=args.nprocs_per_node,
            slowmo_momentum=args.slowmo_momentum,
            localsgd=(args.slowmo_algorithm == 'LocalSGD'),
            localsgd_frequency=args.localsgd_frequency
        )
    else:
        raise ValueError('Unknown --ddp-backend: ' + args.ddp_backend)

    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    ddp_model = _DistributedFairseqModel(**init_kwargs)
    # Assume that:
    # ``args.distributed_wrapper`` is 'DDP' and ``args.ddp_backend`` is 'c10d'.
    
    comm_hook_type = os.getenv("COMM_HOOK_TYPE")
    print(" =================================================== ")
    if comm_hook_type is not None:
        print("DDP communication hook {} is registered".format(comm_hook_type))
    else:
        print("No DDP communication hook is registered.")
    print(" =================================================== ")
    state = powerSGD.PowerSGDState(
        process_group=process_group,
        matrix_approximation_rank=1,
        start_powerSGD_iter=2,  # Use 1_000 if the perplexity becomes worse.
    )
    
    # 1) Register a FP16 compression communication hook.
    if comm_hook_type == "FP16_COMPRESS":
        ddp.register_comm_hook(state=process_group, hook=default.fp16_compress_hook)
    # 2) Register a PowerSGD communication hook.
    elif comm_hook_type == "POWER_SGD":
        ddp_model.register_comm_hook(state, hook=powerSGD.powerSGD_hook)    
    # 3) Register a FP16+PowerSGD communication hook.
    elif comm_hook_type == "FP16_POWER_SGD":
        ddp_model.register_comm_hook(state, default.fp16_compress_wrapper(powerSGD.powerSGD_hook))
    # 4) Register a BatchedPowerSGD communication hook.
    elif comm_hook_type == "BATCHED_POWER_SGD":
        ddp_model.register_comm_hook(state, hook=powerSGD.batched_powerSGD_hook)
    # 4) Register a FP16+BatchedPowerSGD communication hook.
    elif comm_hook_type == "FP16_BATCHED_POWER_SGD":
        ddp_model.register_comm_hook(state, default.fp16_compress_wrapper(powerSGD.batched_powerSGD_hook))
    elif comm_hook_type is not None:
        raise ValueError("Unknown value of the environment variable COMM_HOOK_TYPE.")
    
    return ddp_model


class TPUDistributedDataParallel(nn.Module):

    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = distributed_utils.get_world_size(self.process_group)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        gradients = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if p.grad.requires_grad:
                raise RuntimeError(
                    "TPUDistributedDataParallel only works with gradients that don't "
                    "require grad"
                )
            gradients.append(p.grad)

        import torch_xla.core.xla_model as xm
        xm.all_reduce(
            'sum',
            gradients,
            scale=1. / self.world_size,
            groups=self.process_group[1],
        )
