import time
from collections import defaultdict
from datetime import timedelta
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from codebase.utils import data_utils, model_utils, utils


def train(
    opt: DictConfig, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[int, torch.nn.Module]:
    """
    Train the model.

    Args:
        opt (DictConfig): Configuration options.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        torch.nn.Module: The trained model.
    """
    training_start_time = time.time()

    train_loader = data_utils.get_data(opt, "train")
    step = 0

    while step < opt.training.steps:
        for input_images, labels in train_loader:
            start_time = time.time()
            print_results = (
                opt.training.print_idx > 0 and step % opt.training.print_idx == 0
            )

            input_images = input_images.cuda(non_blocking=True)

            optimizer, lr = utils.update_learning_rate(optimizer, opt, step)
            optimizer.zero_grad()

            loss, metrics = model(input_images, labels, evaluate=print_results)
            loss.backward()

            if opt.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opt.training.gradient_clip
                )

            optimizer.step()

            # Print results.
            if print_results:
                iteration_time = time.time() - start_time
                utils.print_results("train", step, iteration_time, metrics)

            # Validate.
            if opt.training.val_idx > 0 and step % opt.training.val_idx == 0:
                validate_or_test(opt, step, model, "val")

            step += 1
            if step >= opt.training.steps:
                break

    total_train_time = time.time() - training_start_time
    print(f"Total training time: {timedelta(seconds=total_train_time)}")
    return step, model


def validate_or_test(
    opt: DictConfig, step: int, model: torch.nn.Module, partition: str
) -> None:
    """
    Perform validation or testing of the model.

    Args:
        opt (DictConfig): Configuration options.
        step (int): Current training step.
        model (torch.nn.Module): The model to be evaluated.
        partition (str): Partition name ("val" or "test").
    """
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = data_utils.get_data(opt, partition)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            input_images = inputs.cuda(non_blocking=True)

            loss, metrics = model(input_images, labels, evaluate=True)

            test_results["Loss"] += loss.item() / len(data_loader)
            for key, value in metrics.items():
                test_results[key] += value / len(data_loader)

    total_test_time = time.time() - test_time
    utils.print_results(partition, step, total_test_time, test_results)
    model.train()


@hydra.main(config_path="config", config_name="config")
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_opt(opt)

    # Initialize model and optimizer.
    model, optimizer = model_utils.get_model_and_optimizer(opt)

    step, model = train(opt, model, optimizer)
    validate_or_test(opt, step, model, "test")


if __name__ == "__main__":
    my_main()
