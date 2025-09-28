#!/usr/bin/env python3
"""
Evaluation Script for Semantic Diffusion Model
Calculate FID, LPIPS, and semantic consistency metrics

Requirements:
- pytorch-fid
- lpips
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import evaluation libraries
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import transforms
import cv2

class SemanticDiffusionEvaluator:
    """
    Comprehensive evaluator for semantic diffusion model
    """
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Image transformation for evaluation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def calculate_fid_score(self, real_images_path, generated_images_path):
        """
        Calculate Fréchet Inception Distance (FID) score
        """
        print("Calculating FID score...")
        
        try:
            fid_score = calculate_fid_given_paths(
                [str(real_images_path), str(generated_images_path)],
                batch_size=50,
                device=self.device,
                dims=2048
            )
            return fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
            return None
    
    def calculate_lpips_diversity(self, generated_images_dir):
        """
        Calculate LPIPS diversity between generated images
        Higher values indicate more diverse generations
        """
        print("Calculating LPIPS diversity...")
        
        lpips_scores = []
        sample_dirs = list(generated_images_dir.glob('*'))
        
        for sample_dir in tqdm(sample_dirs[:50]):  # Limit for speed
            if not sample_dir.is_dir():
                continue
                
            # Get all generated images for this sample
            gen_images = list(sample_dir.glob('generated_*.png'))
            
            if len(gen_images) < 2:
                continue
            
            # Calculate pairwise LPIPS scores
            sample_scores = []
            for i in range(len(gen_images)):
                for j in range(i + 1, len(gen_images)):
                    img1 = Image.open(gen_images[i]).convert('RGB')
                    img2 = Image.open(gen_images[j]).convert('RGB')
                    
                    # Transform images
                    img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
                    img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
                    
                    # Calculate LPIPS
                    with torch.no_grad():
                        score = self.lpips_model(img1_tensor, img2_tensor).item()
                    sample_scores.append(score)
            
            if sample_scores:
                lpips_scores.extend(sample_scores)
        
        if lpips_scores:
            return {
                'mean': np.mean(lpips_scores),
                'std': np.std(lpips_scores),
                'median': np.median(lpips_scores),
                'scores': lpips_scores
            }
        else:
            return None
    
    def calculate_semantic_consistency(self, data_dir, generated_images_dir):
        """
        Calculate semantic consistency using a pre-trained segmentation model
        This measures how well the generated images match the input semantic layout
        """
        print("Calculating semantic consistency...")
        
        try:
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
            
            # Load pre-trained segmentation model
            feature_extractor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            ).to(self.device)
            model.eval()
            
            consistency_scores = []
            sample_dirs = list(generated_images_dir.glob('*'))
            
            for sample_dir in tqdm(sample_dirs[:20]):  # Limit for speed
                if not sample_dir.is_dir():
                    continue
                
                sample_name = sample_dir.name
                
                # Load original semantic map
                original_label_path = Path(data_dir) / 'labels' / f'{sample_name}.png'
                if not original_label_path.exists():
                    continue
                
                original_label = np.array(Image.open(original_label_path))
                
                # Process generated images
                gen_images = list(sample_dir.glob('generated_*.png'))
                sample_scores = []
                
                for gen_img_path in gen_images:
                    # Load generated image
                    gen_img = Image.open(gen_img_path).convert('RGB')
                    
                    # Predict segmentation
                    inputs = feature_extractor(images=gen_img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    
                    # Upsample predictions to original resolution
                    upsampled_logits = F.interpolate(
                        logits,
                        size=original_label.shape,
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # Get predicted classes
                    predicted = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
                    
                    # Calculate IoU for major classes (simplified)
                    # Note: This is a simplified version - full evaluation would need class mapping
                    intersection = np.logical_and(original_label > 0, predicted > 0).sum()
                    union = np.logical_or(original_label > 0, predicted > 0).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        sample_scores.append(iou)
                
                if sample_scores:
                    consistency_scores.extend(sample_scores)
            
            if consistency_scores:
                return {
                    'mean': np.mean(consistency_scores),
                    'std': np.std(consistency_scores),
                    'median': np.median(consistency_scores),
                    'scores': consistency_scores
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error calculating semantic consistency: {e}")
            print("Skipping semantic consistency evaluation.")
            return None
    
    def calculate_image_quality_metrics(self, real_images_path, generated_images_dir):
        """
        Calculate various image quality metrics
        """
        print("Calculating image quality metrics...")
        
        # Get sample of real and generated images
        real_images = list(Path(real_images_path).glob('*.png'))[:100]
        
        generated_paths = []
        for sample_dir in Path(generated_images_dir).glob('*'):
            if sample_dir.is_dir():
                gen_imgs = list(sample_dir.glob('generated_*.png'))
                generated_paths.extend(gen_imgs[:1])  # Take first generated image per sample
        
        # Calculate basic statistics
        def get_image_stats(image_paths):
            brightness_vals = []
            contrast_vals = []
            
            for img_path in image_paths[:50]:  # Limit for speed
                img = np.array(Image.open(img_path).convert('RGB'))
                
                # Brightness (mean pixel value)
                brightness = np.mean(img)
                brightness_vals.append(brightness)
                
                # Contrast (std of pixel values)
                contrast = np.std(img)
                contrast_vals.append(contrast)
            
            return {
                'brightness_mean': np.mean(brightness_vals),
                'brightness_std': np.std(brightness_vals),
                'contrast_mean': np.mean(contrast_vals),
                'contrast_std': np.std(contrast_vals)
            }
        
        real_stats = get_image_stats(real_images)
        gen_stats = get_image_stats(generated_paths)
        
        return {
            'real_images': real_stats,
            'generated_images': gen_stats
        }
    
    def run_full_evaluation(self, data_dir, generated_images_dir, results_dir):
        """
        Run complete evaluation suite
        """
        print("Starting comprehensive evaluation...")
        
        data_path = Path(data_dir)
        gen_path = Path(generated_images_dir)
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. FID Score
        try:
            real_images_path = data_path / 'images'
            
            # Create temporary directory with all generated images for FID calculation
            temp_gen_dir = results_path / 'temp_generated_flat'
            temp_gen_dir.mkdir(exist_ok=True)
            
            # Flatten generated images directory
            gen_count = 0
            for sample_dir in gen_path.glob('*'):
                if sample_dir.is_dir():
                    for gen_img in sample_dir.glob('generated_*.png'):
                        dest_path = temp_gen_dir / f'gen_{gen_count:06d}.png'
                        # Copy image
                        img = Image.open(gen_img)
                        img.save(dest_path)
                        gen_count += 1
                        
                        if gen_count >= 500:  # Limit for reasonable computation time
                            break
                    if gen_count >= 500:
                        break
            
            fid_score = self.calculate_fid_score(real_images_path, temp_gen_dir)
            if fid_score is not None:
                results['fid_score'] = fid_score
                print(f"FID Score: {fid_score:.2f}")
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_gen_dir)
            
        except Exception as e:
            print(f"Error in FID calculation: {e}")
        
        # 2. LPIPS Diversity
        try:
            lpips_results = self.calculate_lpips_diversity(gen_path)
            if lpips_results:
                results['lpips_diversity'] = lpips_results
                print(f"LPIPS Diversity: {lpips_results['mean']:.4f} ± {lpips_results['std']:.4f}")
        except Exception as e:
            print(f"Error in LPIPS calculation: {e}")
        
        # 3. Semantic Consistency
        try:
            semantic_results = self.calculate_semantic_consistency(data_path, gen_path)
            if semantic_results:
                results['semantic_consistency'] = semantic_results
                print(f"Semantic Consistency: {semantic_results['mean']:.4f} ± {semantic_results['std']:.4f}")
        except Exception as e:
            print(f"Error in semantic consistency calculation: {e}")
        
        # 4. Image Quality Metrics
        try:
            quality_results = self.calculate_image_quality_metrics(
                data_path / 'images', gen_path
            )
            results['image_quality'] = quality_results
            
            print("Image Quality Metrics:")
            print(f"  Real Images - Brightness: {quality_results['real_images']['brightness_mean']:.1f}")
            print(f"  Generated Images - Brightness: {quality_results['generated_images']['brightness_mean']:.1f}")
            print(f"  Real Images - Contrast: {quality_results['real_images']['contrast_mean']:.1f}")
            print(f"  Generated Images - Contrast: {quality_results['generated_images']['contrast_mean']:.1f}")
            
        except Exception as e:
            print(f"Error in quality metrics calculation: {e}")
        
        # Save results
        with open(results_path / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        self.create_evaluation_report(results, results_path / 'evaluation_report.txt')
        
        print(f"\nEvaluation completed! Results saved to {results_path}")
        return results
    
    def create_evaluation_report(self, results, report_path):
        """
        Create a human-readable evaluation report
        """
        with open(report_path, 'w') as f:
            f.write("Semantic Diffusion Model - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            if 'fid_score' in results:
                f.write(f"FID Score: {results['fid_score']:.2f}\n")
                f.write("  (Lower is better - measures image quality and diversity)\n")
                f.write("  Excellent: < 10, Good: 10-30, Fair: 30-50, Poor: > 50\n\n")
            
            if 'lpips_diversity' in results:
                lpips = results['lpips_diversity']
                f.write(f"LPIPS Diversity: {lpips['mean']:.4f} ± {lpips['std']:.4f}\n")
                f.write("  (Higher is better - measures diversity of generations)\n")
                f.write("  Good diversity: > 0.3, Moderate: 0.1-0.3, Low: < 0.1\n\n")
            
            if 'semantic_consistency' in results:
                semantic = results['semantic_consistency']
                f.write(f"Semantic Consistency: {semantic['mean']:.4f} ± {semantic['std']:.4f}\n")
                f.write("  (Higher is better - measures adherence to input semantic layout)\n")
                f.write("  Excellent: > 0.8, Good: 0.6-0.8, Fair: 0.4-0.6, Poor: < 0.4\n\n")
            
            if 'image_quality' in results:
                quality = results['image_quality']
                f.write("Image Quality Comparison:\n")
                f.write(f"  Brightness - Real: {quality['real_images']['brightness_mean']:.1f}, ")
                f.write(f"Generated: {quality['generated_images']['brightness_mean']:.1f}\n")
                f.write(f"  Contrast - Real: {quality['real_images']['contrast_mean']:.1f}, ")
                f.write(f"Generated: {quality['generated_images']['contrast_mean']:.1f}\n\n")
            
            f.write("Evaluation completed successfully!\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Diffusion Model")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to original dataset directory")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Path to directory containing generated images")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        return
    
    if not os.path.exists(args.generated_dir):
        print(f"Error: Generated images directory {args.generated_dir} does not exist!")
        return
    
    # Initialize evaluator
    device = args.device if torch.cuda.is_available() else 'cpu'
    evaluator = SemanticDiffusionEvaluator(device=device)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(
        args.data_dir,
        args.generated_dir,
        args.results_dir
    )
    
    print("\nEvaluation Summary:")
    print("-" * 30)
    
    if 'fid_score' in results:
        print(f"FID Score: {results['fid_score']:.2f}")
    
    if 'lpips_diversity' in results:
        print(f"LPIPS Diversity: {results['lpips_diversity']['mean']:.4f}")
    
    if 'semantic_consistency' in results:
        print(f"Semantic Consistency: {results['semantic_consistency']['mean']:.4f}")
    
    print(f"\nDetailed results saved to: {args.results_dir}")

if __name__ == "__main__":
    main()