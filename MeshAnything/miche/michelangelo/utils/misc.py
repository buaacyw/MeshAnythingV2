# -*- coding: utf-8 -*-

import importlib.util
import importlib, sys

import torch
import torch.distributed as dist

import os
import folder_paths


def list_all_packages():
    return sorted(importlib.metadata.distributions(), key=lambda x: x.metadata["Name"])


def get_obj_from_str(string, reload=False):
    module_name, cls = string.rsplit(".", 1)

    ROOT_PATH = os.path.join(
        folder_paths.base_path,
        "custom_nodes",
        "comfyui_meshanything_v2",
        "MeshAnything",
        "__init__.py",
    )

    spec = importlib.util.spec_from_file_location("MeshAnything", ROOT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["MeshAnything"] = module
    spec.loader.exec_module(module)

    if reload:
        module_imp = importlib.import_module(
            f".{module_name}", "custom_nodes.comfyui_meshanything_v2"
        )
        importlib.reload(module_imp)

    return getattr(
        importlib.import_module(
            f".{module_name}", "custom_nodes.comfyui_meshanything_v2"
        ),
        cls,
    )


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_all, tensor, async_op=False)  # performance opt

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
