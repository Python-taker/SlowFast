#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import functools
import logging
import pickle

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def cat_all_gather(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def init_distributed_training(cfg):
    """
    Initializes distributed training.
    """
    init_process_group(
        local_rank=0,  # assuming single GPU per process; adjust if needed
        local_world_size=cfg.NUM_GPUS,
        shard_id=cfg.SHARD_ID,
        num_shards=cfg.NUM_SHARDS,
        init_method=cfg.INIT_METHOD if hasattr(cfg, 'INIT_METHOD') else "tcp://localhost:9999",
        dist_backend="nccl",
    )


def all_gather(tensors):
    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_reduce(tensors, average=True):
    for tensor in tensors:
        dist.all_reduce(tensor, async_op=False)
    if average:
        world_size = dist.get_world_size()
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def init_process_group(
    local_rank,
    local_world_size,
    shard_id,
    num_shards,
    init_method,
    dist_backend="nccl",
):
    torch.cuda.set_device(local_rank)
    proc_rank = local_rank + shard_id * local_world_size
    world_size = local_world_size * num_shards
    dist.init_process_group(
        backend=dist_backend,
        init_method=init_method,
        world_size=world_size,
        rank=proc_rank,
    )


def is_master_proc(num_gpus=8):
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True


def is_root_proc():
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Returns the world size (total number of processes).
    Defaults to 1 if distributed not initialized.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def synchronize():
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_world_size() == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024**3:
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024**3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather_unaligned(data, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class AllGatherWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        world_size = dist.get_world_size()
        x_gather = [torch.ones_like(input) for _ in range(world_size)]
        torch.distributed.all_gather(x_gather, input, async_op=False)
        x_gather = torch.cat(x_gather, dim=0)
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):
        reduction = torch.distributed.all_reduce(grad_output, async_op=True)
        reduction.wait()

        world_size = dist.get_world_size()
        N = grad_output.size(0)
        mini_batchsize = N // world_size
        cur_gpu = torch.distributed.get_rank()
        grad_output = grad_output[
            cur_gpu * mini_batchsize : (cur_gpu + 1) * mini_batchsize
        ]
        return grad_output
