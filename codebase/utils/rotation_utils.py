from typing import Dict

import torch
from einops import repeat
from omegaconf import DictConfig

from codebase.utils import eval_utils


def add_rotation_dimensions(opt: DictConfig, x: torch.Tensor) -> torch.Tensor:
    """
    Add rotation dimensions to the input tensor. First dimension contains input tensor;
    all following dimensions are set to zero.

    Args:
        opt (DictConfig): Configuration options.
        x (torch.Tensor): The input tensor, shape (b, ...).

    Returns:
        torch.Tensor: The tensor with rotation dimensions added, shape (b, n, ...).
    """

    extra_dimensions = repeat(
        torch.zeros_like(x), "b ... -> b n ...", n=opt.model.rotation_dimensions - 1,
    )
    return torch.cat((x[:, None], extra_dimensions), dim=1)


def rescale_magnitude_rotating_features(
    x: torch.Tensor, scaling_factor: torch.Tensor
) -> torch.Tensor:
    """
    Rescale the magnitude of a rotating features tensor according to scaling_factor.

    Args:
        x (torch.Tensor): The input rotating features tensor, shape (b, n, ...).
        scaling_factor (torch.Tensor): The scaling factor tensor, shape (b, ...).

    Returns:
        torch.Tensor: The scaled tensor, shape (b, n, ...).
    """
    return torch.nn.functional.normalize(x, dim=1) * scaling_factor[:, None]


def run_evaluation(
    opt: DictConfig, rotation_output: torch.Tensor, labels: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Run the evaluation for the reconstructed rotating features. First, normalize and mask the rotating features,
    then cluster, and finally compare predicted labels to the ground-truth.

    Args:
        opt (DictConfig): Configuration options.
        rotation_output (torch.Tensor): The rotating feature output of the model, shape (b, n, c, h, w).
        labels (Dict[str, torch.Tensor]): Dictionary containing ground-truth labels.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    # Map rotating features onto unit-hypersphere, while masking features with small magnitudes.
    norm_rotating_output = norm_and_mask_rotating_output(opt, rotation_output)

    # Cluster rotating features according to their orientation.
    pred_labels = eval_utils.apply_kmeans(
        opt, norm_rotating_output, labels["pixelwise_instance_labels"],
    )

    # Compare predicted cluster labels with ground-truth labels.
    return eval_utils.run_object_discovery_evaluation(opt, pred_labels, labels)


def norm_and_mask_rotating_output(
    opt: DictConfig, rotation_output: torch.Tensor
) -> torch.Tensor:
    """
    Normalize and mask rotating features. First, calculate the magnitude of the rotating features,
    then map them onto the unit hyper-sphere, while masking values below opt.evaluation.magnitude_mask_threshold.
    Finally, take weighted sum of the rotating features across their channel dimension, using the normalized
    and masked magnitudes as weights.

    Args:
        opt (DictConfig): Configuration options.
        rotation_output (torch.Tensor): The rotating features output tensor, shape (b, n, c, h, w).

    Returns:
        torch.Tensor: Rotating features tensor mapped onto the unit hypersphere,
        with values with small magnitudes masked out, shape (b, n, c, h, w).
    """
    magnitude = get_magnitude(rotation_output, dim=1)
    norm_magnitude = norm_and_mask_magnitude(opt, magnitude)
    return get_norm_rotating_output(rotation_output, norm_magnitude)


def get_magnitude(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Calculate the magnitude of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to calculate the magnitude. Default is 1.

    Returns:
        torch.Tensor: The magnitude tensor.
    """
    return torch.linalg.vector_norm(x, dim=dim)


def norm_and_mask_magnitude(opt: DictConfig, magnitude: torch.Tensor) -> torch.Tensor:
    """
    Map magnitudes onto the unit hypersphere while masking out values below
    opt.evaluation.magnitude_mask_threshold.

    Args:
        opt (DictConfig): Configuration options.
        magnitude (torch.Tensor): The magnitude tensor.

    Returns:
        torch.Tensor: The normalized and masked magnitude tensor.
    """
    norm_magnitude = torch.ones_like(magnitude)
    masking_idx = torch.where(magnitude <= opt.evaluation.magnitude_mask_threshold)
    norm_magnitude[masking_idx] = 0
    return norm_magnitude


def get_norm_rotating_output(
    rotating_output: torch.Tensor, norm_magnitude: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Get the normalized rotating features output: map rotating features onto a unit-hypersphere, while masking out
    features with small magnitudes.

    When the input tensor is single-channel, simply scale the rotating features based on the normalized and
    masked magnitudes. When the input tensor is multi-chanel, take the sum of the rotating features across their
    channel dimension, weighted by the normalized and masked magnitudes.

    Args:
        rotating_output (torch.Tensor): The rotating features output tensor, shape (b, n, c, h, w).
        norm_magnitude (torch.Tensor): The normalized and masked magnitude tensor, shape (b, c, h, w).
        eps (float, optional): A small constant for numerical stability. Default is 1e-8.

    Returns:
        torch.Tensor: The normalized rotating features output tensor, shape (b, n, 1, h, w).
    """
    if norm_magnitude.shape[1] == 1:
        # For single-channel images: no weighted sum across channels needed.
        # Simply scale rotating features based on normalized and masked magnitudes.
        return rescale_magnitude_rotating_features(rotating_output, norm_magnitude)

    # Normalize features to lie on unit-sphere.
    normalized_rotating_output = torch.nn.functional.normalize(
        rotating_output, p=2, dim=1
    )

    # Take sum of features across rotation dimension, weighted by normalized and masked magnitudes.
    weighted_sum_rotating_output = torch.sum(
        normalized_rotating_output * norm_magnitude[:, None], dim=2
    ) / (torch.sum(norm_magnitude[:, None], dim=2) + eps)

    return weighted_sum_rotating_output[:, :, None]
