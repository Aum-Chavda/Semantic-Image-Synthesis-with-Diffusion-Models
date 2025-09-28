#!/bin/bash

# Multi-GPU Training Script for Semantic Diffusion Model
# Designed for 8x A100 GPUs on university HPC cluster
# Author: SDM Training Pipeline
# Date: August 2025

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export WORLD_SIZE=8
export NCCL_DEBUG=INFO

# Training parameters
DATA_DIR="./data/sidewalk_processed"
OUTPUT_DIR="./checkpoints/semantic_diffusion_1024"
BATCH_SIZE=2  # Per GPU batch size (total: 2 * 8 = 16)
LEARNING_RATE=1e-4
NUM_EPOCHS=100
MAX_STEPS=50000
GRADIENT_ACCUMULATION=2
MIXED_PRECISION="fp16"

# Model parameters
MODEL_CHANNELS=256
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32 16 8"
CHANNEL_MULT="1 2 4 8"
NUM_HEAD_CHANNELS=64

# System parameters
NUM_WORKERS=8
LOGGING_STEPS=100
SAVE_STEPS=2500

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Log system information
echo "Starting Semantic Diffusion Model Training"
echo "========================================="
echo "Date: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "World Size: $WORLD_SIZE"
echo "Batch Size per GPU: $BATCH_SIZE"
echo "Total Batch Size: $((BATCH_SIZE * WORLD_SIZE))"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Mixed Precision: $MIXED_PRECISION"
echo "Learning Rate: $LEARNING_RATE"
echo "Max Training Steps: $MAX_STEPS"
echo "========================================="

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist!"
    echo "Please run prepare_dataset.py first."
    exit 1
fi

# Check if dataset_info.json exists
if [ ! -f "$DATA_DIR/dataset_info.json" ]; then
    echo "Error: Dataset info file not found!"
    echo "Please run prepare_dataset.py first."
    exit 1
fi

# Log dataset information
echo "Dataset Information:"
echo "==================="
python3 -c "
import json
with open('$DATA_DIR/dataset_info.json', 'r') as f:
    info = json.load(f)
print(f'Total samples: {info[\"processed_samples\"]}')
print(f'Image size: {info[\"image_size\"]}')
print(f'Number of classes: {info[\"num_classes\"]}')
"
echo "==================="

# Start training with accelerate
echo "Starting distributed training..."

accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=1 \
    --mixed_precision=$MIXED_PRECISION \
    --main_process_port=29500 \
    train_semantic_diffusion.py \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=1024 \
    --train_batch_size=$BATCH_SIZE \
    --eval_batch_size=$BATCH_SIZE \
    --num_train_epochs=$NUM_EPOCHS \
    --max_train_steps=$MAX_STEPS \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    --learning_rate=$LEARNING_RATE \
    --lr_warmup_steps=1000 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=1.0 \
    --num_train_timesteps=1000 \
    --model_channels=$MODEL_CHANNELS \
    --num_res_blocks=$NUM_RES_BLOCKS \
    --attention_resolutions $ATTENTION_RESOLUTIONS \
    --channel_mult $CHANNEL_MULT \
    --num_head_channels=$NUM_HEAD_CHANNELS \
    --use_scale_shift_norm \
    --dropout=0.0 \
    --mixed_precision=$MIXED_PRECISION \
    --num_workers=$NUM_WORKERS \
    --logging_steps=$LOGGING_STEPS \
    --save_steps=$SAVE_STEPS \
    --use_wandb \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed!"
echo "Check logs directory for detailed training logs."
echo "Model checkpoints saved in: $OUTPUT_DIR"