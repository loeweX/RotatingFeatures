import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from codebase.data import ShapesDataset, FoodSeg, PascalDataset


def load_dataset(opt: DictConfig, partition: str) -> Dataset:
    """
    Load a dataset based on the configuration.

    Args:
        opt (DictConfig): Configuration options.
        partition (str): Data partition ("train", "val", "test").

    Returns:
        Dataset: The loaded dataset.
    """
    if opt.input.dataset == 0:
        dataset = ShapesDataset.ShapesDataset(opt, partition)
    elif opt.input.dataset == 1:
        dataset = FoodSeg.FoodSeg(opt, partition)
    elif opt.input.dataset == 2:
        dataset = PascalDataset.PascalDataset(opt, partition)
    else:
        raise NotImplementedError("Dataset not implemented.")
    return dataset


def get_dataloader(opt: DictConfig, dataset: Dataset, partition: str) -> DataLoader:
    """
    Get a dataloader for the specified dataset.

    Args:
        opt (DictConfig): Configuration options.
        dataset (Dataset): The dataset to create a data loader for.
        partition (str): Data partition ("train", "val", "test").

    Returns:
        DataLoader: The dataloader.
    """
    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=partition == "train",
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def get_data(opt: DictConfig, partition: str) -> DataLoader:
    """
    Get data for the specified configuration and partition. First loads the dataset,
    then creates and returns a dataloader.

    Args:
        opt (DictConfig): Configuration options.
        partition (str): Data partition ("train", "val", "test").

    Returns:
        DataLoader: The dataloader.
    """
    dataset = load_dataset(opt, partition)
    return get_dataloader(opt, dataset, partition)


def seed_worker(worker_id: int) -> None:
    """
    Seed the data-loading worker for improved reproducibility.

    Args:
        worker_id (int): Worker ID.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_rotate_90(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Randomly rotate the input image tensor by 90 degrees clock- or counterclockwise.

    Args:
        image_tensor (Tensor): Input image tensor, shape (..., h, w).

    Returns:
        Tensor: Potentially rotated image tensor, shape (..., h, w).
    """
    # With 50% chance, do not rotate image.
    if np.random.rand() <= 0.5:
        return image_tensor

    # Randomly choose between -1 (counterclockwise) and 1 (clockwise).
    direction = random.choice([-1, 1])

    # Rotate the image tensor by 90 degrees in the chosen direction.
    return torch.rot90(image_tensor, k=direction, dims=(-2, -1))
