from typing import List

import torch.nn as nn
from omegaconf import DictConfig

from codebase.model import RotatingLayers


class ConvDecoder(nn.Module):
    def __init__(
        self, opt: DictConfig, channel_per_layer: List[int], latent_dim: int,
    ) -> None:
        """
        Initialize the Convolutional Decoder.

        Args:
            opt (DictConfig): Configuration options.
            channel_per_layer (List[int]): List of channel sizes per layer.
            latent_dim (int): Size of the latent dimension.
        """
        super().__init__()

        # Initialize convolutional layers.
        self.convolutional = nn.ModuleList()
        for i in range(2, -1, -1):
            self.convolutional.append(
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[i + 1],
                    channel_per_layer[i + 1],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                )  # Scales up features map size, e.g. from 4x4 to 8x8.
            )

            self.convolutional.append(
                RotatingLayers.RotatingConv2d(
                    opt,
                    channel_per_layer[i + 1],
                    channel_per_layer[i],
                    kernel_size=3,
                    padding=1,
                )
            )

        # Initialize linear layer.
        self.linear = RotatingLayers.RotatingLinear(
            opt, opt.model.linear_dim, latent_dim
        )
