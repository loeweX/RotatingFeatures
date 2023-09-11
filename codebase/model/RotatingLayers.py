import math

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor

from codebase.utils import rotation_utils


def apply_layer_to_rotating_features(
    opt: DictConfig,
    layer: nn.Module,
    rotation_bias: nn.Parameter,
    norm: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a layer to the input rotating features tensor.

    Args:
        opt (DictConfig): Configuration options.
        layer (nn.Module): Layer to apply (Conv2d, ConvTranspose2d, or Linear).
        rotation_bias (nn.Parameter): Rotation bias parameter.
        norm (nn.Module): Normalization layer.
        x (torch.Tensor): Input rotating features tensor, shape: (b, n, c, h, w) or (b, n, c).

    Returns:
        torch.Tensor: Output rotating features tensor, shape: (b, n, c, h, w) or (b, n, c).
    """
    if isinstance(layer, torch.nn.Linear):
        psi = layer(x)  # (b, n, c).
    elif isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        # Fold rotation dimension into batch dimension.
        x_folded = rearrange(x, "b n c h w -> (b n) c h w")
        psi_folded = layer(x_folded)
        psi = rearrange(psi_folded, "(b n) c h w -> b n c h w", b=opt.input.batch_size)
    else:
        raise NotImplementedError

    magnitude_psi = rotation_utils.get_magnitude(psi)

    # Apply rotation bias.
    z = psi + rotation_bias

    # Apply binding mechanism.
    chi = layer(rotation_utils.get_magnitude(x))
    magnitude = 0.5 * magnitude_psi + 0.5 * chi

    # Apply activation function.
    magnitude = norm(magnitude)
    magnitude = nn.functional.relu(magnitude)
    return rotation_utils.rescale_magnitude_rotating_features(z, magnitude)


def init_rotation_bias(fan_in: int, bias: nn.Parameter) -> nn.Parameter:
    """
    Initialize the rotation bias parameter.

    Args:
        fan_in (int): Fan-in value.
        bias (nn.Parameter): Bias parameter to be initialized.

    Returns:
        nn.Parameter: Initialized bias parameter.
    """
    bound = 1 / math.sqrt(fan_in)
    return torch.nn.init.uniform_(bias, -bound, bound)


class RotatingConvTranspose2d(nn.Module):
    def __init__(
        self,
        opt: DictConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ) -> None:
        """
        Initialize a 2D transposed convolution layer with rotating features.

        Args:
            opt (DictConfig): Configuration options.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to the input. Default is 0.
            output_padding (int, optional): Additional padding added to the output. Default is 0.
        """
        super(RotatingConvTranspose2d, self).__init__()

        self.opt = opt
        self.conv_tran = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )
        self.fan_in = (
            out_channels * self.conv_tran.kernel_size[0] * self.conv_tran.kernel_size[1]
        )

        self.rotation_bias = nn.Parameter(
            torch.empty((1, opt.model.rotation_dimensions, out_channels, 1, 1))
        )
        self.rotation_bias = init_rotation_bias(self.fan_in, self.rotation_bias)

        self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 2D transposed convolution layer with rotating features.

        Args:
            x (Tensor): Input tensor, shape: (b, n, c, h, w).

        Returns:
            Tensor: Output tensor, shape: (b, n, c, h, w).
        """
        return apply_layer_to_rotating_features(
            self.opt, self.conv_tran, self.rotation_bias, self.norm, x
        )


class RotatingConv2d(nn.Module):
    def __init__(
        self,
        opt: DictConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        """
        Initialize a 2D convolution layer with rotating features.

        Args:
            opt (DictConfig): Configuration options.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            stride (int, optional): Stride of the convolution. Default is 1.
            padding (int, optional): Padding added to the input. Default is 0.
        """
        super(RotatingConv2d, self).__init__()

        self.opt = opt
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False,
        )
        self.fan_in = in_channels * self.conv.kernel_size[0] * self.conv.kernel_size[1]

        self.rotation_bias = nn.Parameter(
            torch.empty((1, opt.model.rotation_dimensions, out_channels, 1, 1))
        )
        self.rotation_bias = init_rotation_bias(self.fan_in, self.rotation_bias)

        self.norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 2D convolution layer with rotating features.

        Args:
            x (Tensor): Input tensor, shape: (b, n, c, h, w).

        Returns:
            Tensor: Output tensor, shape: (b, n, c, h, w).
        """
        return apply_layer_to_rotating_features(
            self.opt, self.conv, self.rotation_bias, self.norm, x
        )


class RotatingLinear(nn.Module):
    def __init__(self, opt: DictConfig, in_features: int, out_features: int) -> None:
        """
        Initialize a linear layer with rotating features.

        Args:
            opt (DictConfig): Configuration options.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(RotatingLinear, self).__init__()

        self.opt = opt
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.fan_in = in_features

        self.rotation_bias = nn.Parameter(
            torch.empty((1, opt.model.rotation_dimensions, out_features))
        )
        self.rotation_bias = init_rotation_bias(self.fan_in, self.rotation_bias)

        self.norm = nn.LayerNorm(out_features, elementwise_affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer with rotating features.

        Args:
            x (Tensor): Input tensor, shape: (b, n, c).

        Returns:
            Tensor: Output tensor, shape: (b, n, c).
        """
        return apply_layer_to_rotating_features(self.opt, self.fc, self.rotation_bias, self.norm, x)
