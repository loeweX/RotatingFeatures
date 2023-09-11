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
        self.convolutional = nn.ModuleList(
            [
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[3],
                    channel_per_layer[3],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                ),  # Scales up features map size, e.g. from 4x4 to 8x8.
                RotatingLayers.RotatingConv2d(
                    opt,
                    channel_per_layer[3],
                    channel_per_layer[2],
                    kernel_size=3,
                    padding=1,
                ),
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[2],
                    channel_per_layer[2],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                ),
                RotatingLayers.RotatingConv2d(
                    opt,
                    channel_per_layer[2],
                    channel_per_layer[1],
                    kernel_size=3,
                    padding=1,
                ),
            ]
        )

        if opt.input.dino_processed:
            self.convolutional.append(
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[1],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                )
            )
            self.convolutional.append(
                RotatingLayers.RotatingConv2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[0],
                    kernel_size=3,
                    padding=1,
                ),
            )
        else:
            self.convolutional.append(
                RotatingLayers.RotatingConvTranspose2d(
                    opt,
                    channel_per_layer[1],
                    channel_per_layer[0],
                    kernel_size=3,
                    output_padding=1,
                    padding=1,
                    stride=2,
                )
            )

        # Initialize linear layer.
        self.linear = RotatingLayers.RotatingLinear(
            opt, opt.model.linear_dim, latent_dim
        )
