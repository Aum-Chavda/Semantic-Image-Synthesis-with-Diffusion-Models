# Semantic Diffusion Model Training - Complete Documentation

## Overview

This repository contains a complete implementation for training a **Semantic Diffusion Model** on the HuggingFace sidewalk dataset, converting it to 1024×1024 resolution with colored semantic masks, and generating 5 new images per sample using multiple A100 GPUs.

Based on the paper: **"Semantic Image Synthesis via Diffusion Models"** by Wang et al.

## System Requirements

### Hardware
- **GPUs**: 8× NVIDIA A100 80GB (university HPC cluster)
- **Memory**: 64GB+ system RAM
- **Storage**: 1TB+ fast SSD storage
- **Network**: High-bandwidth connection for dataset download

### Software
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 11.8+
- **Python**: 3.9
- **PyTorch**: 2.0+
- **Conda/Miniconda**: Latest version

## Project Structure

```
semantic-diffusion-sidewalk/
├── data/
│   └── sidewalk_processed/          # Processed dataset (1024×1024)
│       ├── images/                  # RGB images
│       ├── labels/                  # Semantic masks (grayscale)
│       ├── colored_masks/           # Colored semantic masks
│       └── dataset_info.json       # Dataset metadata
├── checkpoints/
│   └── semantic_diffusion_1024/     # Model checkpoints
│       ├── final_model.pth          # Trained model
│       └── finetuned/               # Fine-tuned model (CFG)
├── results/
│   └── generated/                   # Generated images
│       ├── generated_images/        # Individual samples
│       └── visualizations/          # Comparison grids
├── evaluation_results/              # Evaluation metrics
├── logs/                           # Training logs
├── scripts/                        # Main scripts
│   ├── prepare_dataset.py          # Dataset preparation
│   ├── train_semantic_diffusion.py # Training script
│   ├── generate_images.py          # Generation script
│   ├── evaluate_model.py           # Evaluation script
│   ├── train_multi_gpu.sh          # Multi-GPU training
│   └── run_complete_pipeline.sh    # Full pipeline
├── requirements.txt                # Dependencies
├── setup_environment.sh            # Environment setup
└── README.md                       # This file
```

## Dataset Information

### HuggingFace Sidewalk Dataset
- **Source**: `segments/sidewalk-semantic`
- **Classes**: 35 semantic classes
- **Original Resolution**: Variable
- **Processed Resolution**: 1024×1024
- **Total Samples**: ~1000+ images

### Class Mapping (35 classes)
```
0: unlabeled          18: construction-building
1: flat-road          19: construction-door
2: flat-sidewalk      20: construction-wall
3: flat-crosswalk     21: construction-fenceguardrail
4: flat-cyclinglane   22: construction-bridge
5: flat-parkingdriveway 23: construction-tunnel
6: flat-railtrack     24: construction-stairs
7: flat-curb          25: object-pole
8: human-person       26: object-trafficsign
9: human-rider        27: object-trafficlight
10: vehicle-car       28: nature-vegetation
11: vehicle-truck     29: nature-terrain
12: vehicle-bus       30: sky
13: vehicle-tramtrain 31: void-ground
14: vehicle-motorcycle 32: void-dynamic
15: vehicle-bicycle   33: void-static
16: vehicle-caravan   34: void-unclear
17: vehicle-cartrailer
```

## Installation and Setup

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd semantic-diffusion-sidewalk

# Make scripts executable
chmod +x *.sh scripts/*.sh

# Run environment setup
./setup_environment.sh

# Activate environment
conda activate sdm-training
```

### 2. Verify Installation
```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check GPU status
nvidia-smi
```

## Usage

### Option 1: Complete Pipeline (Recommended)
```bash
# Run entire pipeline automatically
./run_complete_pipeline.sh
```

This script will:
1. Setup environment
2. Download and process dataset
3. Train the model
4. Fine-tune for classifier-free guidance
5. Generate images
6. Evaluate results

### Option 2: Step-by-Step Execution

#### Step 1: Dataset Preparation
```bash
python3 prepare_dataset.py \
    --output_dir ./data/sidewalk_processed \
    --num_workers 8
```

#### Step 2: Model Training
```bash
# Multi-GPU training (8 GPUs)
./train_multi_gpu.sh

# Or manual training
accelerate launch --multi_gpu --num_processes=8 \
    train_semantic_diffusion.py \
    --data_dir ./data/sidewalk_processed \
    --output_dir ./checkpoints/semantic_diffusion_1024 \
    --train_batch_size 2 \
    --max_train_steps 50000 \
    --learning_rate 1e-4
```

#### Step 3: Image Generation
```bash
python3 generate_images.py \
    --model_path ./checkpoints/semantic_diffusion_1024/final_model.pth \
    --data_dir ./data/sidewalk_processed \
    --output_dir ./results/generated \
    --num_samples_per_image 5 \
    --guidance_scale 1.5
```

#### Step 4: Evaluation
```bash
python3 evaluate_model.py \
    --data_dir ./data/sidewalk_processed \
    --generated_dir ./results/generated/generated_images \
    --results_dir ./evaluation_results
```

## Model Architecture

### Semantic Diffusion Model Components

#### 1. **U-Net Backbone**
- **Base Channels**: 256
- **Channel Multipliers**: [1, 2, 4, 8]
- **Attention Layers**: At 32×32, 16×16, 8×8 resolutions
- **ResNet Blocks**: 2 per resolution level

#### 2. **SPADE Normalization**
- Spatially-Adaptive Normalization in decoder
- Semantic map conditions normalization parameters
- Preserves semantic information throughout denoising

#### 3. **Timestep Embedding**
- Sinusoidal position encoding
- Injected into encoder via scale-shift normalization

#### 4. **Classifier-Free Guidance**
- Fine-tuning with 20% semantic mask dropout
- Guidance scale: 1.5 (configurable)

### Training Configuration

#### Initial Training
- **Optimizer**: AdamW (lr=1e-4, β₁=0.9, β₂=0.999)
- **Scheduler**: Cosine with warmup (1000 steps)
- **Batch Size**: 16 (2 per GPU × 8 GPUs)
- **Steps**: 50,000
- **Mixed Precision**: FP16
- **Gradient Clipping**: 1.0

#### Fine-tuning (Classifier-Free Guidance)
- **Optimizer**: AdamW (lr=2e-5)
- **Dropout Rate**: 0.2
- **Steps**: 10,000
- **Purpose**: Enable conditional/unconditional generation

### Diffusion Parameters
- **Timesteps**: 1000 (training), 50 (sampling)
- **Noise Schedule**: Linear (β₁=0.0001, β₁=0.02)
- **Prediction Type**: ε-prediction (noise)

## Performance Specifications

### Training Performance (8× A100 80GB)
- **Memory Usage**: ~60GB per GPU
- **Training Time**: ~24-48 hours (50k steps)
- **Fine-tuning Time**: ~8-12 hours (10k steps)
- **Throughput**: ~16 samples/second

### Generation Performance
- **Generation Time**: ~30-60 seconds per image (50 steps)
- **Memory Usage**: ~40GB
- **Batch Generation**: 5 images simultaneously

### Expected Results
- **FID Score**: 15-25 (lower is better)
- **LPIPS Diversity**: 0.3-0.5 (higher is better)
- **Semantic Consistency**: 0.6-0.8 (higher is better)

## File Descriptions

### Core Scripts

#### `prepare_dataset.py`
- Downloads HuggingFace sidewalk dataset
- Resizes images to 1024×1024
- Creates colored semantic masks
- Generates dataset statistics

#### `train_semantic_diffusion.py`
- Main training script
- Implements SPADE-based U-Net
- Supports distributed training
- Integrates with Weights & Biases

#### `generate_images.py`
- Image generation from trained models
- Supports classifier-free guidance
- Batch generation capabilities
- Creates visualization grids

#### `evaluate_model.py`
- Comprehensive evaluation suite
- FID, LPIPS, semantic consistency
- Generates evaluation reports

### Utility Scripts

#### `train_multi_gpu.sh`
- Multi-GPU training orchestration
- Environment variable setup
- Logging and monitoring

#### `run_complete_pipeline.sh`
- End-to-end pipeline execution
- Error handling and recovery
- Progress tracking

## Evaluation Metrics

### 1. Fréchet Inception Distance (FID)
- **Purpose**: Measures image quality and diversity
- **Range**: 0-∞ (lower better)
- **Target**: < 30 for good results

### 2. LPIPS Diversity
- **Purpose**: Measures generation diversity
- **Range**: 0-1 (higher better)
- **Target**: > 0.3 for good diversity

### 3. Semantic Consistency
- **Purpose**: Adherence to input semantic layout
- **Method**: Segmentation model evaluation
- **Target**: > 0.6 for good consistency

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce batch size
export BATCH_SIZE=1

# Enable gradient checkpointing
export USE_CHECKPOINT=true

# Use CPU offloading
export OFFLOAD_TO_CPU=true
```

#### NCCL Communication Errors
```bash
# Set NCCL debug
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=300

# Use different backend
export NCCL_BACKEND=gloo
```

#### Dataset Access Issues
```bash
# Login to HuggingFace
huggingface-cli login

# Request dataset access
# Visit: https://huggingface.co/datasets/segments/sidewalk-semantic
```

### Performance Optimization

#### Memory Optimization
- Use gradient accumulation: `--gradient_accumulation_steps 4`
- Enable mixed precision: `--mixed_precision fp16`
- Use gradient checkpointing: `--use_checkpoint`

#### Speed Optimization
- Increase number of workers: `--num_workers 16`
- Use faster storage (NVMe SSD)
- Enable compilation: `torch.compile(model)`

## Remote GPU Access Setup

### SSH Configuration
```bash
# Connect to university cluster
ssh username@cluster.university.edu

# Set up port forwarding for monitoring
ssh -L 8080:localhost:8080 username@cluster.university.edu
```

### SLURM Job Submission
```bash
# Create SLURM script
cat > submit_training.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=semantic_diffusion
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --time=72:00:00
#SBATCH --mem=512GB
#SBATCH --cpus-per-task=64

module load cuda/11.8
source ~/.bashrc
conda activate sdm-training

./run_complete_pipeline.sh
EOF

# Submit job
sbatch submit_training.sbatch
```

### Monitoring
```bash
# Check job status
squeue -u username

# Monitor GPU usage
nvidia-smi -l 1

# View logs
tail -f logs/training_*.log

# Check Weights & Biases
# Visit: https://wandb.ai
```

## Results and Output

### Generated Files
- **Training Logs**: `logs/training_*.log`
- **Model Checkpoints**: `checkpoints/semantic_diffusion_1024/`
- **Generated Images**: `results/generated/generated_images/`
- **Visualizations**: `results/generated/visualizations/`
- **Evaluation Report**: `evaluation_results/evaluation_report.txt`

### Sample Output Structure
```
results/generated/generated_images/
├── sample_000001/
│   ├── generated_00.png
│   ├── generated_01.png
│   ├── generated_02.png
│   ├── generated_03.png
│   └── generated_04.png
├── sample_000002/
│   └── ...
```

## Citation

If you use this implementation, please cite:

```bibtex
@article{wang2022semantic,
  title={Semantic Image Synthesis via Diffusion Models},
  author={Wang, Weilun and Bao, Jianmin and Zhou, Wengang and Chen, Dongdong and Chen, Dong and Yuan, Lu and Li, Houqiang},
  journal={arXiv preprint arXiv:2207.00050},
  year={2022}
}
```

## License

This project is for research and educational purposes. Please respect the licenses of the original dataset and model implementations.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for specific errors
3. Verify system requirements and setup
4. Consult original paper and HuggingFace documentation

## Acknowledgments

- Original Semantic Diffusion Model authors
- HuggingFace for the sidewalk dataset
- University HPC cluster for computational resources
- PyTorch and diffusers library developers