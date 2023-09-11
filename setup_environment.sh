#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create conda environment.
conda env create -f environment.yml
source activate RotatingFeatures

# Install additional packages.
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install hydra-core=1.1.0
pip install einops=0.4.1
pip install timm

