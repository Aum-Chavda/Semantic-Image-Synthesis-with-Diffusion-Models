#!/bin/bash

# Environment Setup for Semantic Diffusion Training on Multiple A100 GPUs
# Author: SDM Training Pipeline
# Date: August 2025

echo "Setting up Semantic Diffusion Model Training Environment..."

# Create conda environment
conda create -n sdm-training python=3.9 -y
conda activate sdm-training

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install additional dependencies
pip install -r requirements.txt

# Install Hugging Face libraries
pip install transformers datasets accelerate diffusers

# Install image processing libraries
pip install opencv-python pillow scikit-image
pip install matplotlib seaborn

# Install distributed training libraries
pip install mpi4py

# Set environment variables for multi-GPU training
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=8  # Number of A100 GPUs
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create necessary directories
mkdir -p data/sidewalk_processed
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

echo "Environment setup completed!"
echo "Run: conda activate sdm-training"
echo "Then execute the training pipeline."