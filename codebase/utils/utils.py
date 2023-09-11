import random
from datetime import timedelta
from typing import Dict

import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import open_dict, OmegaConf, DictConfig


def parse_opt(opt: DictConfig) -> DictConfig:
    """
    Parse configuration options and set the random seed.

    Args:
        opt (DictConfig): Configuration options.

    Returns:
        DictConfig: Updated configuration options.
    """
    with open_dict(opt):
        opt.cwd = get_original_cwd()

        if "add_depth_channel" in opt.input and opt.input.add_depth_channel:
            opt.input.channel += 1

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_learning_rate(opt: DictConfig, step: int, lr: float) -> float:
    """
    Get the current learning rate according to the learning rate schedule set in the configuration options.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    if opt.training.learning_rate_schedule == 0:
        return lr
    elif opt.training.learning_rate_schedule == 1:
        return get_linear_warmup_lr(opt, step, lr)
    else:
        raise NotImplementedError


def get_linear_warmup_lr(opt: DictConfig, step: int, lr: float) -> float:
    """
    Calculate the linear warm-up learning rate.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        lr (float): Initial learning rate.

    Returns:
        float: Updated learning rate.
    """
    if step < opt.training.warmup_steps:
        return lr * step / opt.training.warmup_steps
    else:
        return lr


def update_learning_rate(optimizer, opt: DictConfig, step: int):
    """
    Update the learning rate of the optimizer.

    Args:
        optimizer: The optimizer.
        opt (DictConfig): Configuration options.
        step (int): Current training step.

    Returns:
        optimizer: Updated optimizer.
        lr (float): Updated learning rate.
    """
    lr = get_learning_rate(opt, step, opt.training.learning_rate)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def print_results(
    partition: str, step: int, iteration_time: float, metrics: Dict[str, float]
):
    """
    Print training or evaluation results.

    Args:
        partition (str): Partition name (e.g., "train" or "val").
        step (int): Current training step.
        iteration_time (float): Time taken for the iteration.
        metrics (Dict[str, float]): Dictionary of evaluation metrics.
    """
    print(
        f"{partition} \t \t"
        f"Step: {step} \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if metrics is not None:
        for key, value in metrics.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def tensor_dict_to_numpy(
    tensor_dict: Dict[str, torch.Tensor], dtype=np.float32
) -> Dict[str, np.ndarray]:
    """
    Convert a dictionary of PyTorch tensors into a dictionary of NumPy arrays.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): A dictionary of PyTorch tensors.
        dtype (Type[np.ndarray], optional): Data type for the resulting NumPy arrays. Default is np.float32.

    Returns:
        Dict[str, np.ndarray]: A dictionary of NumPy arrays.
    """
    for key in tensor_dict:
        tensor_dict[key] = tensor_dict[key].detach().cpu().numpy().astype(dtype)
    return tensor_dict
