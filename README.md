Semantic Image Synthesis with Diffusion Models

Author: Aum Chavda | LinkedIn 
Project Overview

This project is a complete implementation of the "Semantic Image Synthesis via Diffusion Models" paper, focused on generating high-resolution, photorealistic images from semantic segmentation masks. The primary goal is to leverage a state-of-the-art diffusion model to augment computer vision datasets, thereby improving the performance and generalization of segmentation models.

The entire pipeline, from data preparation to distributed training and evaluation, has been built to run on a multi-GPU High-Performance Computing (HPC) cluster.
Key Features

    High-Resolution Synthesis: Generates 1024x1024 pixel images that are semantically consistent with input masks.

    Dataset Augmentation: Trained on the HuggingFace Sidewalk dataset to create novel training examples.

    State-of-the-Art Architecture: Implements a U-Net backbone with Spatially-Adaptive (SPADE) normalization to inject semantic information.

    Distributed Training: Utilizes PyTorch accelerate for efficient multi-GPU training (8x NVIDIA A100).

    Classifier-Free Guidance: Employs fine-tuning with semantic mask dropout to improve image quality and mask adherence.

    Comprehensive Evaluation: Measures performance using standard metrics like Fréchet Inception Distance (FID), LPIPS, and Semantic Consistency.

Technology Stack

    Core Framework: Python, PyTorch

    Libraries: Diffusers, Accelerate, NumPy, OpenCV, scikit-image

    Dataset: HuggingFace Datasets (segments/sidewalk-semantic)

    Hardware/Platform: NVIDIA A100 GPUs, SLURM, Conda

Project Pipeline

The project follows a systematic 4-step pipeline:

    Data Preparation: The script prepare_dataset.py downloads the raw dataset, resizes images and masks to 1024x1024, and creates colored masks for visualization.

    Model Training: The core training logic is in train_semantic_diffusion.py, orchestrated for multi-GPU execution using train_multi_gpu.sh. The model is trained for 50,000 steps, followed by a fine-tuning phase for classifier-free guidance.

    Image Generation: Using the trained checkpoints, generate_images.py takes unseen semantic masks and generates multiple photorealistic images for each.

    Evaluation: evaluate_model.py calculates key metrics to quantitatively assess the quality, diversity, and semantic consistency of the generated images.

Challenges & Key Learnings

    Model Selection: The initial phase involved a deep dive into current literature to select the most suitable architecture. The Semantic Diffusion Model was chosen for its novel use of SPADE normalization in a diffusion context.

    Architectural Implementation: Translating the complex U-Net with SPADE from the research paper into stable, functional PyTorch code was a significant implementation challenge.

    HPC Management: Managing the training pipeline on a SLURM-based HPC cluster required robust scripting for job submission, resource allocation (memory and GPU), and debugging of distributed training errors (e.g., NCCL timeouts).

How to Run
1. Setup Environment

Clone the repository and set up the Conda environment.

# Clone the project
git clone <repository-url>
cd <repository-directory>

# Create and activate the conda environment
conda env create -f environment.yml
conda activate sdm-training

2. Prepare the Dataset

Download and process the dataset from HuggingFace.

python3 scripts/prepare_dataset.py --output_dir ./data/sidewalk_processed

3. Train the Model

Launch the distributed training script (configured for 8 GPUs).

./scripts/train_multi_gpu.sh

4. Generate Images

Use the trained model to generate new images from the test set masks.

python3 scripts/generate_images.py \
    --model_path ./checkpoints/semantic_diffusion_1024/final_model.pth \
    --data_dir ./data/sidewalk_processed \
    --output_dir ./results/generated
