import torch.nn as nn
from omegaconf import DictConfig

from codebase.model import RotatingLayers
from codebase.utils import model_utils


class ConvEncoder(nn.Module):
    def __init__(self, opt: DictConfig) -> None:
        """
        Initialize the Convolutional Encoder.

        Args:
            opt (DictConfig): Configuration options.
        """
        super().__init__()

        self.channel_per_layer = [
            opt.input.channel,
            opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
            2 * opt.model.hidden_dim,
        ]

        # Initialize convolutional layers.
        self.convolutional = nn.ModuleList(
            [
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[0],
                    self.channel_per_layer[1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),  # Scales down features map size, e.g. from 8x8 to 4x4.
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[1],
                    self.channel_per_layer[1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[1],
                    self.channel_per_layer[2],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[2],
                    self.channel_per_layer[2],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                ),
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[2],
                    self.channel_per_layer[3],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                ),
            ]
        )

        if opt.input.dino_processed:
            self.convolutional.append(
                RotatingLayers.RotatingConv2d(
                    opt,
                    self.channel_per_layer[3],
                    self.channel_per_layer[3],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                )
            )

        # Initialize linear layer.
        self.latent_feature_map_size, self.latent_dim = model_utils.get_latent_dim(
            opt, self.channel_per_layer[-1]
        )
        self.linear = RotatingLayers.RotatingLinear(
            opt, self.latent_dim, opt.model.linear_dim,
        )
