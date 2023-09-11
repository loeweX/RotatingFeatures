from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from codebase.model import ConvDecoder, ConvEncoder
from codebase.utils import rotation_utils, model_utils


class RotatingAutoEncoder(nn.Module):
    def __init__(self, opt: DictConfig) -> None:
        """
        Initialize the RotatingAutoEncoder.

        We implement Rotating Features within an autoencoding architecture. This class contains the entire
        architecture, including the encoder and decoder, the potential preprocessing model, and the output model.
        It also implements the forward pass and evaluation of the model.

        Args:
            opt (DictConfig): Configuration options.
        """
        super(RotatingAutoEncoder, self).__init__()

        self.opt = opt

        # Create model.
        self.encoder = ConvEncoder.ConvEncoder(opt)
        self.decoder = ConvDecoder.ConvDecoder(
            opt, self.encoder.channel_per_layer, self.encoder.latent_dim,
        )

        if self.opt.input.dino_processed:
            # Load DINO model and BN preprocess model that goes with it.
            self.dino = model_utils.load_dino_model()
            self.preprocess_model = nn.Sequential(
                nn.BatchNorm2d(self.opt.input.channel), nn.ReLU(),
            )

        # Create output model.
        self.output_weight = nn.Parameter(torch.empty(self.opt.input.channel))
        self.output_bias = nn.Parameter(torch.empty(1, self.opt.input.channel, 1, 1))
        nn.init.constant_(self.output_weight, 1)
        nn.init.constant_(self.output_bias, 0)

    def preprocess_input_images(
        self, input_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess input images.

        Args:
            input_images (torch.Tensor): Batch of input images, shape (b, c, h, w).

        Returns:
            tuple: Tuple containing preprocessed images to process, shape (b, n, c, h, w),
            and to reconstruct, shape (b, c, h, w), by model.
        """
        if self.opt.input.dino_processed:
            with torch.no_grad():
                dino_features = self.dino(input_images)
                dino_features = rearrange(
                    dino_features,
                    "b (h w) c -> b c h w",
                    h=self.opt.input.image_size[0],
                    w=self.opt.input.image_size[1],
                )
            images_to_process = self.preprocess_model(dino_features)
            images_to_reconstruct = dino_features
        else:
            images_to_process = input_images
            images_to_reconstruct = input_images

        if torch.min(images_to_process) < 0:
            raise AssertionError("images_to_process has to be positive valued.")

        images_to_process = rotation_utils.add_rotation_dimensions(
            self.opt, images_to_process
        )
        return images_to_process, images_to_reconstruct

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Encode the input image.

        Args:
            z (torch.Tensor): Input rotating features tensor, shape (b, n, c, h, w).

        Returns:
            torch.Tensor: Encoded rotating features tensor, shape (b, n, c).
        """
        for layer in self.encoder.convolutional:
            z = layer(z)

        z = rearrange(z, "... c h w -> ... (c h w)")
        z = self.encoder.linear(z)
        return z

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the encoded input image.

        Args:
            z (torch.Tensor): Encoded tensor, shape (b, n, c).

        Returns:
            tuple: Tuple containing reconstructed tensor with rotation dimensions, shape (b, n, c, h, w),
             and reconstructed image, shape (b, c, h, w).
        """
        z = self.decoder.linear(z)

        z = rearrange(
            z,
            "... (c h w) -> ... c h w",
            c=self.encoder.channel_per_layer[-1],
            h=self.encoder.latent_feature_map_size[0],
            w=self.encoder.latent_feature_map_size[1],
        )

        for layer in self.decoder.convolutional:
            z = layer(z)

        reconstruction = self.apply_output_model(rotation_utils.get_magnitude(z))

        rotation_output, reconstruction = self.center_crop_reconstruction(
            z, reconstruction
        )
        return rotation_output, reconstruction

    def apply_output_model(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the output model. This model is applied to the magnitude of the reconstructed rotating features
        tensor and learns an appropriate scaling and shift of the magnitudes to better match the input values.

        Args:
            z (torch.Tensor): Magnitude of reconstructed tensor with rotation dimensions, shape (b, c, h, w).

        Returns:
            torch.Tensor: Reconstructed image, shape (b, c, h, w).
        """
        reconstruction = (
            torch.einsum("b c h w, c -> b c h w", z, self.output_weight)
            + self.output_bias
        )

        if self.opt.input.dino_processed:
            return reconstruction
        else:
            return torch.sigmoid(reconstruction)

    def center_crop_reconstruction(
        self, rotation_output: torch.Tensor, reconstruction: torch.Tensor
    ) -> tuple:
        """
        Center crop the reconstructions as necessary to match the input image size.

        Args:
            rotation_output (torch.Tensor): Reconstructed tensor with rotation dimensions, shape (b, n, c, h, w).
            reconstruction (torch.Tensor): Reconstructed image, shape (b, c, h, w).

        Returns:
            tuple: Tuple containing center-cropped reconstructions.
        """
        if self.opt.input.dino_processed:
            rotation_output = rotation_output[:, :, :, 1:-1, 1:-1]
            reconstruction = reconstruction[:, :, 1:-1, 1:-1]
        return rotation_output, reconstruction

    def forward(
        self, input_images: torch.Tensor, labels: dict, evaluate: bool = False
    ) -> tuple:
        """
        Forward pass through the model, including preprocessing, autoencoding, and evaluation.

        Args:
            input_images (torch.Tensor): Input images, shape (b, c, h, w).
            labels (dict): Labels.
            evaluate (bool): Flag to evaluate or not.

        Returns:
            tuple: Tuple containing loss and other metrics.
        """
        # Prepare input.
        images_to_process, images_to_reconstruct = self.preprocess_input_images(
            input_images
        )

        # Encode & decode.
        z = self.encode(images_to_process)
        rotation_output, reconstruction = self.decode(z)

        # Calculate loss.
        loss = nn.functional.mse_loss(reconstruction, images_to_reconstruct)

        if evaluate:
            # Run evaluation.
            metrics = rotation_utils.run_evaluation(self.opt, rotation_output, labels)
        else:
            metrics = {}

        return loss, metrics
