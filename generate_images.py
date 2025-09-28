#!/usr/bin/env python3
"""
Semantic Diffusion Model Image Generation Script
Generate 5 new images per sample from the sidewalk dataset using trained model

Based on: "Semantic Image Synthesis via Diffusion Models" (Wang et al.)
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import diffusion components
from diffusers import DDPMScheduler
from train_semantic_diffusion import SemanticUNet2D, SidewalkDataset

class SemanticDiffusionGenerator:
    """
    Generator class for semantic diffusion model
    """
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load model configuration and weights
        self.model = SemanticUNet2D(
            in_channels=3,
            out_channels=3,
            semantic_nc=35,
            model_channels=256,
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            channel_mult=[1, 2, 4, 8],
            num_head_channels=64,
            use_scale_shift_norm=True,
            dropout=0.0
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Initialize scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # Set scheduler for inference
        self.scheduler.set_timesteps(50)  # 50 sampling steps
    
    @torch.no_grad()
    def generate_images(self, semantic_map, num_samples=5, guidance_scale=1.5, 
                       generator=None):
        """
        Generate images conditioned on semantic map
        
        Args:
            semantic_map: Tensor of shape (1, 35, H, W) - one-hot semantic map
            num_samples: Number of images to generate
            guidance_scale: Classifier-free guidance scale
            generator: Random number generator for reproducibility
            
        Returns:
            List of generated images as PIL Images
        """
        batch_size = num_samples
        height, width = semantic_map.shape[-2:]
        
        # Duplicate semantic map for batch
        semantic_map_batch = semantic_map.repeat(batch_size, 1, 1, 1)
        
        # Prepare unconditional semantic map (all zeros) for classifier-free guidance
        unconditional_map = torch.zeros_like(semantic_map_batch)
        
        # Concatenate conditional and unconditional inputs
        semantic_input = torch.cat([semantic_map_batch, unconditional_map])
        
        # Generate random noise
        shape = (batch_size, 3, height, width)
        noise = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        
        # Duplicate noise for conditional and unconditional predictions
        noise_input = torch.cat([noise, noise])
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Generating images"):
            # Duplicate timestep
            timestep_batch = t.unsqueeze(0).repeat(batch_size * 2).to(self.device)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(noise_input, timestep_batch, semantic_input)
            
            # Perform classifier-free guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Compute previous sample
            noise_input = self.scheduler.step(noise_pred, t, noise[:batch_size]).prev_sample
            
            # Duplicate for next iteration
            noise_input = torch.cat([noise_input, noise_input])
        
        # Convert to images
        images = []
        final_samples = noise_input[:batch_size]
        
        for i in range(batch_size):
            # Convert from [-1, 1] to [0, 1]
            image = (final_samples[i] + 1) / 2
            image = torch.clamp(image, 0, 1)
            
            # Convert to PIL
            image_np = image.cpu().permute(1, 2, 0).numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            images.append(image_pil)
        
        return images
    
    @torch.no_grad()
    def generate_with_fine_tuning_guidance(self, semantic_map, num_samples=5, 
                                         guidance_scale=1.5, generator=None):
        """
        Generate images with classifier-free guidance (requires fine-tuned model)
        """
        return self.generate_images(semantic_map, num_samples, guidance_scale, generator)

def load_color_palette():
    """Load the color palette for visualization"""
    return [
        [0, 0, 0],          # 0: unlabeled - black
        [128, 64, 128],     # 1: flat-road - purple
        [244, 35, 232],     # 2: flat-sidewalk - magenta
        [70, 70, 70],       # 3: flat-crosswalk - dark gray
        [102, 102, 156],    # 4: flat-cyclinglane - blue-gray
        [190, 153, 153],    # 5: flat-parkingdriveway - light brown
        [153, 153, 153],    # 6: flat-railtrack - gray
        [250, 170, 30],     # 7: flat-curb - orange
        [220, 220, 0],      # 8: human-person - yellow
        [107, 142, 35],     # 9: human-rider - olive
        [0, 0, 142],        # 10: vehicle-car - blue
        [0, 0, 70],         # 11: vehicle-truck - dark blue
        [0, 60, 100],       # 12: vehicle-bus - navy
        [0, 80, 100],       # 13: vehicle-tramtrain - dark cyan
        [0, 0, 230],        # 14: vehicle-motorcycle - bright blue
        [119, 11, 32],      # 15: vehicle-bicycle - maroon
        [0, 0, 90],         # 16: vehicle-caravan - very dark blue
        [0, 0, 110],        # 17: vehicle-cartrailer - darker blue
        [70, 130, 180],     # 18: construction-building - steel blue
        [20, 20, 255],      # 19: construction-door - bright blue
        [102, 102, 156],    # 20: construction-wall - blue-gray
        [190, 153, 153],    # 21: construction-fenceguardrail - light brown
        [153, 153, 153],    # 22: construction-bridge - gray
        [128, 128, 128],    # 23: construction-tunnel - medium gray
        [250, 170, 160],    # 24: construction-stairs - light orange
        [153, 153, 153],    # 25: object-pole - gray
        [220, 220, 0],      # 26: object-trafficsign - yellow
        [250, 170, 30],     # 27: object-trafficlight - orange
        [107, 142, 35],     # 28: nature-vegetation - olive
        [152, 251, 152],    # 29: nature-terrain - light green
        [70, 130, 180],     # 30: sky - steel blue
        [220, 20, 60],      # 31: void-ground - crimson
        [255, 0, 0],        # 32: void-dynamic - red
        [0, 0, 255],        # 33: void-static - blue
        [255, 255, 255]     # 34: void-unclear - white
    ]

def create_visualization_grid(original_image, semantic_map_colored, generated_images, 
                            sample_name, save_path):
    """
    Create a visualization grid showing original image, semantic map, and generated images
    """
    n_generated = len(generated_images)
    fig, axes = plt.subplots(2, max(3, (n_generated + 2) // 2), figsize=(15, 8))
    fig.suptitle(f'Sample: {sample_name}', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Original image
    axes_flat[0].imshow(original_image)
    axes_flat[0].set_title('Original Image')
    axes_flat[0].axis('off')
    
    # Semantic map
    axes_flat[1].imshow(semantic_map_colored)
    axes_flat[1].set_title('Semantic Map')
    axes_flat[1].axis('off')
    
    # Generated images
    for i, gen_img in enumerate(generated_images):
        axes_flat[i + 2].imshow(gen_img)
        axes_flat[i + 2].set_title(f'Generated {i + 1}')
        axes_flat[i + 2].axis('off')
    
    # Hide remaining subplots
    for i in range(len(generated_images) + 2, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def semantic_map_to_colored(semantic_map_onehot, palette):
    """
    Convert one-hot semantic map to colored visualization
    """
    # Convert one-hot to class indices
    semantic_indices = torch.argmax(semantic_map_onehot, dim=0).cpu().numpy()
    
    # Create colored image
    colored = np.zeros((*semantic_indices.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        mask = (semantic_indices == class_id)
        colored[mask] = color
    
    return colored

def generate_from_dataset(args):
    """
    Generate images for all samples in the dataset
    """
    print(f"Loading model from {args.model_path}...")
    
    # Initialize generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = SemanticDiffusionGenerator(args.model_path, device)
    
    print(f"Loading dataset from {args.data_dir}...")
    
    # Load dataset
    dataset = SidewalkDataset(args.data_dir, split='val')  # Use validation set
    
    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'generated_images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
    
    # Load color palette
    palette = load_color_palette()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    print(f"Generating {args.num_samples_per_image} images for {len(dataset)} samples...")
    
    # Process each sample
    for idx in tqdm(range(min(args.max_samples, len(dataset))), desc="Processing samples"):
        sample = dataset[idx]
        
        # Get data
        original_image = sample['image']  # CHW tensor
        semantic_map = sample['semantic_map']  # CHW one-hot tensor
        filename = sample['filename']
        
        # Convert original image to PIL for visualization
        original_pil = Image.fromarray((original_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        # Create colored semantic map for visualization
        semantic_colored = semantic_map_to_colored(semantic_map, palette)
        
        # Prepare semantic map for generation (add batch dimension)
        semantic_input = semantic_map.unsqueeze(0).to(device)
        
        # Generate images
        try:
            generated_images = generator.generate_images(
                semantic_input,
                num_samples=args.num_samples_per_image,
                guidance_scale=args.guidance_scale
            )
            
            # Save individual generated images
            sample_dir = output_dir / 'generated_images' / filename
            sample_dir.mkdir(exist_ok=True)
            
            for i, gen_img in enumerate(generated_images):
                gen_img.save(sample_dir / f'generated_{i:02d}.png')
            
            # Create and save visualization
            visualization_path = output_dir / 'visualizations' / f'{filename}_grid.png'
            create_visualization_grid(
                original_pil,
                semantic_colored,
                generated_images,
                filename,
                visualization_path
            )
            
            print(f"Processed {filename}: generated {len(generated_images)} images")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Generation completed! Results saved to {output_dir}")
    print(f"Individual images: {output_dir / 'generated_images'}")
    print(f"Visualizations: {output_dir / 'visualizations'}")

def main():
    parser = argparse.ArgumentParser(description="Generate images using trained Semantic Diffusion Model")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to processed dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results/generated",
                        help="Output directory for generated images")
    
    # Generation parameters
    parser.add_argument("--num_samples_per_image", type=int, default=5,
                        help="Number of images to generate per semantic map")
    parser.add_argument("--guidance_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of dataset samples to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        return
    
    # Start generation
    generate_from_dataset(args)

if __name__ == "__main__":
    main()