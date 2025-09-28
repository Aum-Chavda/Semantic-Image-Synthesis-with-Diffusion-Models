#!/usr/bin/env python3
"""
Sidewalk Dataset Preparation for Semantic Diffusion Model Training
Converts HuggingFace sidewalk dataset to 1024x1024 with colored masks

Dataset Classes (35 total):
0-unlabeled, 1-flat-road, 2-flat-sidewalk, 3-flat-crosswalk, 4-flat-cyclinglane,
5-flat-parkingdriveway, 6-flat-railtrack, 7-flat-curb, 8-human-person, 9-human-rider,
10-vehicle-car, 11-vehicle-truck, 12-vehicle-bus, 13-vehicle-tramtrain, 14-vehicle-motorcycle,
15-vehicle-bicycle, 16-vehicle-caravan, 17-vehicle-cartrailer, 18-construction-building,
19-construction-door, 20-construction-wall, 21-construction-fenceguardrail, 22-construction-bridge,
23-construction-tunnel, 24-construction-stairs, 25-object-pole, 26-object-trafficsign,
27-object-trafficlight, 28-nature-vegetation, 29-nature-terrain, 30-sky, 31-void-ground,
32-void-dynamic, 33-void-static, 34-void-unclear
"""

import os
import numpy as np
import torch
from PIL import Image
import cv2
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial

# Color palette for 35 classes (RGB values)
SIDEWALK_PALETTE = [
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

def create_colored_mask(label_array, palette=SIDEWALK_PALETTE):
    """
    Convert single-channel semantic mask to RGB colored mask
    
    Args:
        label_array: numpy array of shape (H, W) with class indices
        palette: list of RGB colors for each class
    
    Returns:
        colored_mask: numpy array of shape (H, W, 3) with RGB colors
    """
    colored_mask = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(palette):
        mask = (label_array == class_id)
        colored_mask[mask] = color
    
    return colored_mask

def resize_with_aspect_ratio(image, target_size=1024):
    """
    Resize image to target_size x target_size while maintaining aspect ratio
    Pad with zeros if necessary
    """
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    if len(image.shape) == 3:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create padded image
    if len(image.shape) == 3:
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=image.dtype)
    
    # Calculate padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Place resized image in center
    if len(image.shape) == 3:
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized
    else:
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return padded

def process_single_sample(args):
    """Process a single sample from the dataset"""
    sample_data, idx, output_dir = args
    
    try:
        # Extract image and label
        image = sample_data['pixel_values']
        label = sample_data['label']
        
        # Convert PIL to numpy if necessary
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        if hasattr(label, 'convert'):
            label = np.array(label.convert('L'))
        
        # Ensure correct data types
        image = np.array(image, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)
        
        # Resize to 1024x1024
        image_resized = resize_with_aspect_ratio(image, 1024)
        label_resized = resize_with_aspect_ratio(label, 1024)
        
        # Create colored mask
        colored_mask = create_colored_mask(label_resized, SIDEWALK_PALETTE)
        
        # Save files
        img_path = output_dir / 'images' / f'{idx:06d}.png'
        label_path = output_dir / 'labels' / f'{idx:06d}.png'
        colored_path = output_dir / 'colored_masks' / f'{idx:06d}.png'
        
        # Save image, label, and colored mask
        Image.fromarray(image_resized).save(img_path)
        Image.fromarray(label_resized, mode='L').save(label_path)
        Image.fromarray(colored_mask).save(colored_path)
        
        return f"Processed sample {idx}"
        
    except Exception as e:
        return f"Error processing sample {idx}: {str(e)}"

def prepare_sidewalk_dataset(output_dir='./data/sidewalk_processed', num_workers=8):
    """
    Download and preprocess the Hugging Face sidewalk dataset
    Convert to 1024x1024 resolution with colored masks
    """
    print("Loading Hugging Face sidewalk dataset...")
    
    # Load dataset (requires authentication)
    try:
        dataset = load_dataset("segments/sidewalk-semantic", split="train")
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have access to segments/sidewalk-semantic dataset")
        return False
    
    # Create output directories
    output_path = Path(output_dir)
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    (output_path / 'colored_masks').mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(dataset)} samples to {output_dir}...")
    
    # Prepare arguments for multiprocessing
    process_args = [(dataset[i], i, output_path) for i in range(len(dataset))]
    
    # Process samples in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_sample, process_args),
            total=len(dataset),
            desc="Processing samples"
        ))
    
    # Print results
    success_count = sum(1 for r in results if not r.startswith("Error"))
    print(f"Successfully processed {success_count}/{len(dataset)} samples")
    
    # Save dataset statistics
    stats = {
        'total_samples': len(dataset),
        'processed_samples': success_count,
        'image_size': '1024x1024',
        'num_classes': 35,
        'class_names': [
            'unlabeled', 'flat-road', 'flat-sidewalk', 'flat-crosswalk', 'flat-cyclinglane',
            'flat-parkingdriveway', 'flat-railtrack', 'flat-curb', 'human-person', 'human-rider',
            'vehicle-car', 'vehicle-truck', 'vehicle-bus', 'vehicle-tramtrain', 'vehicle-motorcycle',
            'vehicle-bicycle', 'vehicle-caravan', 'vehicle-cartrailer', 'construction-building',
            'construction-door', 'construction-wall', 'construction-fenceguardrail', 'construction-bridge',
            'construction-tunnel', 'construction-stairs', 'object-pole', 'object-trafficsign',
            'object-trafficlight', 'nature-vegetation', 'nature-terrain', 'sky', 'void-ground',
            'void-dynamic', 'void-static', 'void-unclear'
        ],
        'palette': SIDEWALK_PALETTE
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset preparation completed!")
    print(f"Images: {output_path / 'images'}")
    print(f"Labels: {output_path / 'labels'}")
    print(f"Colored masks: {output_path / 'colored_masks'}")
    print(f"Dataset info: {output_path / 'dataset_info.json'}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare sidewalk dataset for semantic diffusion training')
    parser.add_argument('--output_dir', type=str, default='./data/sidewalk_processed',
                        help='Output directory for processed dataset')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers for processing')
    
    args = parser.parse_args()
    
    success = prepare_sidewalk_dataset(args.output_dir, args.num_workers)
    
    if success:
        print("Dataset preparation completed successfully!")
    else:
        print("Dataset preparation failed!")
        exit(1)