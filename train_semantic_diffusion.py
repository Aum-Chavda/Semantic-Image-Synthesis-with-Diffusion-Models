#!/usr/bin/env python3
"""
Semantic Diffusion Model Training Script
Multi-GPU training on A100 cluster for sidewalk dataset

Based on: "Semantic Image Synthesis via Diffusion Models" (Wang et al.)
Architecture: U-Net with Spatially-Adaptive Normalization (SPADE)
"""

import os
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
from tqdm import tqdm
import wandb
from accelerate import Accelerator

# Import diffusion components
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

class SPADEResnetBlock(nn.Module):
    """
    ResNet block with Spatially-Adaptive Normalization (SPADE)
    """
    def __init__(self, fin, fout, semantic_nc, kernel_size=3, padding=1):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        
        # Create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size, padding=padding)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size, padding=padding)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        
        # SPADE normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)
    
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        
        out = x_s + dx
        return out
    
    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s
    
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    """
    Spatially-Adaptive Normalization
    """
    def __init__(self, norm_nc, label_nc, kernel_size=3, padding=1):
        super().__init__()
        
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        
        # The dimension of the intermediate embedding space
        nhidden = 128
        
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size, padding=padding),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size, padding=padding)
    
    def forward(self, x, segmap):
        # Part 1. Generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        
        # Part 2. Produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap.float(), size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        
        # Apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out

class SemanticUNet2D(nn.Module):
    """
    U-Net with SPADE normalization for semantic conditioning
    """
    def __init__(self, 
                 in_channels=3,
                 out_channels=3, 
                 semantic_nc=35,
                 model_channels=256,
                 num_res_blocks=2,
                 attention_resolutions=[32, 16, 8],
                 channel_mult=[1, 2, 4, 8],
                 num_head_channels=64,
                 use_scale_shift_norm=True,
                 dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.semantic_nc = semantic_nc
        self.model_channels = model_channels
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    TimestepEmbedResnetBlock(
                        ch, mult * model_channels, time_embed_dim, 
                        dropout=dropout, use_scale_shift_norm=use_scale_shift_norm
                    )
                )
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    self.encoder_blocks.append(
                        AttentionBlock(ch, num_head_channels)
                    )
            
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(Downsample(ch))
                ds *= 2
        
        # Middle block
        self.middle_block = nn.Sequential(
            TimestepEmbedResnetBlock(
                ch, ch, time_embed_dim, dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(ch, num_head_channels),
            TimestepEmbedResnetBlock(
                ch, ch, time_embed_dim, dropout=dropout,
                use_scale_shift_norm=use_scale_shift_norm
            )
        )
        
        # Decoder blocks with SPADE
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_ch = mult * model_channels if i == num_res_blocks else ch
                self.decoder_blocks.append(
                    SPADEResnetBlock(
                        ch + skip_ch, mult * model_channels, semantic_nc
                    )
                )
                ch = mult * model_channels
                
                if ds in attention_resolutions:
                    self.decoder_blocks.append(
                        AttentionBlock(ch, num_head_channels)
                    )
            
            if level != 0:
                self.decoder_blocks.append(Upsample(ch))
                ds //= 2
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, timesteps, semantic_map=None):
        # Time embedding
        t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Encoder
        h = self.input_conv(x)
        hs = [h]
        
        for block in self.encoder_blocks:
            if hasattr(block, 'time_embed'):
                h = block(h, t_emb)
            else:
                h = block(h)
            hs.append(h)
        
        # Middle
        for block in self.middle_block:
            if hasattr(block, 'time_embed'):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Decoder with SPADE conditioning
        for block in self.decoder_blocks:
            if isinstance(block, SPADEResnetBlock):
                skip_h = hs.pop()
                h = torch.cat([h, skip_h], dim=1)
                h = block(h, semantic_map)
            elif hasattr(block, 'time_embed'):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        return self.output_conv(h)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepEmbedResnetBlock(nn.Module):
    """
    ResNet block with timestep embedding
    """
    def __init__(self, in_channels, out_channels, temb_channels, 
                 dropout=0.0, use_scale_shift_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_embed = nn.Linear(temb_channels, out_channels * 2 if use_scale_shift_norm else out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        temb_out = self.time_embed(F.silu(temb))
        temb_out = temb_out[:, :, None, None]
        
        if self.use_scale_shift_norm:
            scale, shift = temb_out.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = self.norm2(h + temb_out)
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_connection(x)

class AttentionBlock(nn.Module):
    """
    Multi-head attention block
    """
    def __init__(self, channels, num_head_channels):
        super().__init__()
        self.channels = channels
        self.num_heads = channels // num_head_channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, h * w))
        qkv = qkv.view(b, 3 * c, self.num_heads, h * w // self.num_heads)
        h = self.attention(qkv)
        h = self.proj_out(h.view(b, c, h * w))
        return (x + h.view(b, c, h, w)).contiguous()

class QKVAttention(nn.Module):
    """
    QKV attention implementation
    """
    def forward(self, qkv):
        bs, width, heads, length = qkv.shape
        assert width % (3 * heads) == 0
        ch = width // (3 * heads)
        
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(ch)
        
        weight = torch.einsum("bchq,bchk->bhqk", q * scale, k)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        a = torch.einsum("bhqk,bchk->bchq", weight, v)
        return a.contiguous().view(bs, -1, length)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SidewalkDataset(Dataset):
    """
    Dataset loader for processed sidewalk data
    """
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load dataset info
        with open(self.data_dir / 'dataset_info.json', 'r') as f:
            self.info = json.load(f)
        
        # Get all image files
        img_dir = self.data_dir / 'images'
        self.image_files = sorted(list(img_dir.glob('*.png')))
        
        # Split dataset (80% train, 20% val)
        n_train = int(0.8 * len(self.image_files))
        if split == 'train':
            self.image_files = self.image_files[:n_train]
        else:
            self.image_files = self.image_files[n_train:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem
        
        # Load image and semantic mask
        image = Image.open(img_path).convert('RGB')
        label_path = self.data_dir / 'labels' / f'{img_name}.png'
        label = Image.open(label_path).convert('L')
        
        # Convert to tensors
        image = torch.tensor(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        label = torch.tensor(np.array(label)).long()
        
        # Convert label to one-hot encoding
        semantic_map = F.one_hot(label, num_classes=35).float()
        semantic_map = semantic_map.permute(2, 0, 1)  # HWC -> CHW
        
        if self.transform:
            image = self.transform(image)
            semantic_map = self.transform(semantic_map)
        
        return {
            'image': image,
            'semantic_map': semantic_map,
            'filename': img_name
        }

def train_semantic_diffusion(args):
    """
    Main training function for semantic diffusion model
    """
    # Initialize distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        project_dir=args.output_dir
    )
    
    # Initialize wandb
    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project="semantic-diffusion-sidewalk",
            name=f"sdm-{args.resolution}-{args.num_train_epochs}epochs",
            config=vars(args)
        )
    
    # Create model
    model = SemanticUNet2D(
        in_channels=3,
        out_channels=3,
        semantic_nc=35,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        channel_mult=args.channel_mult,
        num_head_channels=args.num_head_channels,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout
    )
    
    # Initialize scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    
    # Create datasets
    train_dataset = SidewalkDataset(
        args.data_dir, 
        split='train'
    )
    
    val_dataset = SidewalkDataset(
        args.data_dir,
        split='val'
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        model.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_images = batch['image']
                semantic_maps = batch['semantic_map']
                
                # Sample noise
                noise = torch.randn_like(clean_images)
                bsz = clean_images.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=clean_images.device
                ).long()
                
                # Add noise to clean images
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
                # Predict noise residual
                model_pred = model(noisy_images, timesteps, semantic_maps)
                
                # Compute loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                global_step += 1
                
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch
                    }
                    
                    if args.use_wandb and accelerator.is_main_process:
                        wandb.log(logs)
                    
                    accelerator.print(f"Step {global_step}: loss={loss:.4f}, lr={logs['lr']:.2e}")
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        model_unwrapped = accelerator.unwrap_model(model)
        torch.save(model_unwrapped.state_dict(), 
                   os.path.join(args.output_dir, "final_model.pth"))
    
    accelerator.end_training()

def main():
    parser = argparse.ArgumentParser(description="Train Semantic Diffusion Model")
    
    # Model parameters
    parser.add_argument("--model_channels", type=int, default=256)
    parser.add_argument("--num_res_blocks", type=int, default=2) 
    parser.add_argument("--attention_resolutions", nargs="+", type=int, default=[32, 16, 8])
    parser.add_argument("--channel_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num_head_channels", type=int, default=64)
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    
    # Training parameters
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    
    # System parameters
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_semantic_diffusion(args)

if __name__ == "__main__":
    main()