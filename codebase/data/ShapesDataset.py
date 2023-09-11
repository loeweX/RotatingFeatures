from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset


class ShapesDataset(Dataset):
    def __init__(self, opt: DictConfig, partition: str) -> None:
        """
        Initialize the 4Shapes datasets.

        This dataset class can load the following versions of the 4Shapes dataset:

        The 4Shapes dataset comprises grayscale images of dimensions 32 x 32, each containing four distinct
        white shapes (square, up/downward facing triangle, circle) on a black background. The dataset consists
        of 50,000 images in the train set, and 10,000 images for the validation and test sets, respectively.
        All pixel values fall within the range [0,1].

        The 4Shapes RGB(-D) datasets follow the same general setup, but randomly samples the color of each shape.
        To create the RGB-D variant, we incorporate a depth channel to each image and assign a unique depth value
        within the range [0,1] to every object, maintaining equal distances between them.
        By changing the configuration options, it is possible to choose the number of colors used throughout the
        dataset (opt.input.num_rand_colors) and whether the depth channel is included (opt.input.add_depth_channel).

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(ShapesDataset, self).__init__()

        self.opt = opt
        self.root_dir = Path(opt.cwd, opt.input.load_path)
        self.partition = partition

        if opt.input.file_name == "4Shapes_RGBD":
            if opt.input.add_depth_channel:
                file_name = Path(
                    self.root_dir,
                    f"{opt.input.file_name}_depth_{opt.input.num_rand_colors}_{partition}.npz",
                )
            else:
                file_name = Path(
                    self.root_dir,
                    f"{opt.input.file_name}_{opt.input.num_rand_colors}_{partition}.npz",
                )
        else:
            file_name = Path(self.root_dir, f"{opt.input.file_name}_{partition}.npz")

        dataset = np.load(file_name)
        self.images = dataset["images"].astype(np.float32)
        self.pixelwise_instance_labels = dataset["labels"]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """
        input_image = self.images[idx]
        labels = {"pixelwise_instance_labels": self.pixelwise_instance_labels[idx]}
        return input_image, labels
