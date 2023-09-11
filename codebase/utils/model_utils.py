import math
from typing import Tuple

import timm
import torch
from omegaconf import DictConfig

from codebase.model import RotatingAutoEncoder


def get_model_and_optimizer(
    opt: DictConfig,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Create and return the RotatingAutoEncoder model and its optimizer.

    Args:
        opt (DictConfig): Configuration options.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer]: A tuple containing the model and optimizer.
    """
    model = RotatingAutoEncoder.RotatingAutoEncoder(opt)
    model = model.cuda()

    print(model, "\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.training.learning_rate,
        weight_decay=opt.training.weight_decay,
    )

    return model, optimizer


def load_dino_model() -> torch.nn.Module:
    """
    Load and return the DINO pre-trained model.

    Returns:
        torch.nn.Module: The DINO pre-trained model.
    """
    dino = timm.create_model("hf_hub:timm/vit_base_patch16_224.dino", pretrained=True)
    # Remove last three layers.
    dino = torch.nn.Sequential(*list(dino.children())[:-3])
    dino.requires_grad_(False)
    return dino


def get_latent_dim(
    opt: DictConfig, latent_channels: int
) -> Tuple[Tuple[int, int], int]:
    """
    Calculate and return the latent feature map size and dimension.

    Args:
        opt (DictConfig): Configuration options.
        latent_channels (int): The number of latent channels.

    Returns:
        Tuple[Tuple[int, int], int]: A tuple containing the height and width of the latent feature map
        and its dimension.
    """
    height = math.ceil(opt.input.image_size[0] / (2 ** 3))
    width = math.ceil(opt.input.image_size[1] / (2 ** 3))
    latent_dim = height * width * latent_channels
    return (height, width), latent_dim
