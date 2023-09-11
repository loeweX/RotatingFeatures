import os
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from omegaconf import DictConfig
from torch.utils.data import Dataset

from codebase.utils import data_utils


class FoodSeg(Dataset):
    def __init__(self, opt: DictConfig, partition: str) -> None:
        """
        Initialize the FoodSeg103 dataset.
        See https://xiongweiwu.github.io/foodseg103.html for more information about the dataset.

        Args:
            opt (DictConfig): Configuration options.
            partition (str): Dataset partition ("train", "val", or "test").
        """
        super(FoodSeg, self).__init__()

        self.opt = opt
        self.partition = partition

        # Set image and annotation directories based on configuration.
        self.img_dir = Path(opt.cwd, opt.input.load_path, "img_dir", partition)
        self.img_files = [
            name for name in os.listdir(self.img_dir) if name.endswith(".jpg")
        ]
        self.ann_dir = Path(opt.cwd, opt.input.load_path, "ann_dir", partition)

        # Initialize image processing operations.
        self.to_tensor = transforms.ToTensor()
        self.image_size = 224
        self.random_crop = transforms.Compose(
            [transforms.RandomCrop(self.image_size), transforms.RandomHorizontalFlip()]
        )
        self.center_crop = transforms.CenterCrop(self.image_size)
        self.image_resize = transforms.Resize(
            self.image_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # For evaluation on validation and test set: use a larger label size.
        self.eval_size = 320
        self.label_center_crop = transforms.CenterCrop(self.eval_size)
        self.label_resize = transforms.Resize(
            self.eval_size, interpolation=transforms.InterpolationMode.NEAREST
        )

        # Normalize input images using mean and standard deviation of ImageNet.
        self.normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        """
        Get the number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input image and corresponding gt_labels.
        """
        input_image = self.to_tensor(PILImage.open(self.img_dir / self.img_files[idx]))
        input_image = self.normalize(input_image)

        pixelwise_class_label = self.to_tensor(
            PILImage.open(self.ann_dir / self.img_files[idx].replace("jpg", "png"))
        )
        pixelwise_class_label = (pixelwise_class_label * 255).int()

        # Ensure input and label shapes match.
        if (
            input_image.shape[-2] != pixelwise_class_label.shape[-2]
            or input_image.shape[-1] != pixelwise_class_label.shape[-1]
        ):
            input_image = torch.nn.functional.interpolate(
                input_image[None],
                size=(pixelwise_class_label.shape[-2], pixelwise_class_label.shape[-1]),
                mode="bicubic",
            )[0]

        if "train" in self.partition:
            # Apply random transforms to both inputs and gt_labels for training.
            all_inputs = torch.cat((input_image, pixelwise_class_label), 0)
            all_inputs = self.random_crop(self.image_resize(all_inputs))

            if self.opt.input.random_rotation:
                all_inputs = data_utils.random_rotate_90(all_inputs)

            input_image, pixelwise_class_label = (
                all_inputs[:3],
                all_inputs[-1],
            )
        else:
            # Apply center crop for validation and test datasets.
            input_image = self.center_crop(self.image_resize(input_image))
            pixelwise_class_label = self.label_center_crop(
                self.label_resize(pixelwise_class_label)
            )[0]

        labels = {
            "pixelwise_class_labels": pixelwise_class_label,
            "pixelwise_instance_labels": torch.zeros_like(pixelwise_class_label),
        }
        return input_image, labels
