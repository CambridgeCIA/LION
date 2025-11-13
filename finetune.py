#!/usr/bin/env python3
"""
Finetune Swin2SR for Physically Interpretable CT Reconstruction

This script demonstrates how to finetune the Hugging Face Swin2SR model
to produce physically meaningful CT values in Hounsfield Units (HU).

Key features:
- Preserves CT units during training
- Uses supervised learning with clean CT targets
- Implements proper normalization strategies
- Saves checkpoints and metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import os
import sys

# LION imports
from LION.models.CNNs.huggingface import Swin2SR
from LION.optimizers.SupervisedSolver import SupervisedSolver
from LION.experiments.ct_experiments import LowDoseCTRecon
from LION.utils.parameter import LIONParameter
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM


def setup_distributed():
    """Initialize distributed training if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank_zero(rank):
    """Check if this is the rank 0 process."""
    return rank == 0


class PhysicallyAwareSwin2SR(Swin2SR):
    """
    Swin2SR with modifications for physically interpretable outputs
    """
    
    def __init__(self, geometry, model_parameters=None):
        super().__init__(geometry, model_parameters)
        
        # Add output scaling layer to ensure proper HU range
        self.output_scaling = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        """Forward pass with physical constraints"""
        # Get base Swin2SR output
        base_output = super().forward(x)
        
        # Apply learnable scaling and bias to maintain HU range
        # This helps the model learn to output values in the correct HU range
        scaled_output = base_output * self.output_scaling + self.output_bias
        
        return scaled_output


class CTNormalizer:
    """
    Custom normalizer that preserves CT units while enabling stable training
    """
    
    def __init__(self, hu_min=-1000, hu_max=2000):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.range = hu_max - hu_min
        
    def normalize(self, x):
        """Normalize to [0, 1] while preserving relative HU relationships"""
        return (x - self.hu_min) / self.range
    
    def denormalize(self, x):
        """Convert back to HU units"""
        return x * self.range + self.hu_min


def create_finetuning_dataset(experiment, normalizer, device):
    """
    Create training dataset with proper normalization
    """
    print("Creating finetuning dataset...")
    
    # Get training data
    train_dataset = experiment.get_training_dataset()
    val_dataset = experiment.get_validation_dataset()
    
    # Create custom dataset that handles normalization
    class CTFinetuningDataset:
        def __init__(self, base_dataset, normalizer, device):
            self.base_dataset = base_dataset
            self.normalizer = normalizer
            self.device = device
            
        def __len__(self):
            return len(self.base_dataset)
            
        def __getitem__(self, idx):
            sinogram, ground_truth = self.base_dataset[idx]
            
            # Move to device
            sinogram = sinogram.to(self.device)
            ground_truth = ground_truth.to(self.device)
            
            # Create FDK reconstruction (input to model)
            from LION.classical_algorithms.fdk import fdk
            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)

            # Normalize tensor shape to (C, H, W) for a single sample
            # fdk() may return (H, W), (C, H, W) or (1, H, W). We want to
            # ensure the per-sample tensor has a channel dimension but no batch dim.
            if fdk_recon.dim() == 2:
                # (H, W) -> (1, H, W)
                fdk_recon = fdk_recon.unsqueeze(0)
            elif fdk_recon.dim() == 4 and fdk_recon.size(0) == 1:
                # Sometimes fdk might return a singleton-batched tensor (1, C, H, W)
                # -> squeeze to (C, H, W)
                fdk_recon = fdk_recon.squeeze(0)
            # If it's already (C, H, W) we keep it as-is. If it has unexpected
            # number of dims (e.g. 5) we'll let the downstream assertion catch it
            # and raise a clear error.
            
            # Normalize both input and target for stable training
            fdk_normalized = normalizer.normalize(fdk_recon)
            target_normalized = normalizer.normalize(ground_truth)
            
            return fdk_normalized, target_normalized
    
    train_finetuning = CTFinetuningDataset(train_dataset, normalizer, device)
    val_finetuning = CTFinetuningDataset(val_dataset, normalizer, device)
    
    return train_finetuning, val_finetuning


def finetune_swin2sr(
    model_name="caidas/swin2SR-classical-sr-x2-64",
    dataset="LIDC-IDRI",
    epochs=100,
    batch_size=1,
    learning_rate=1e-3,
    save_dir="finetuned_swin2sr",
    device=None,
    use_amp=True,
    accumulation_steps=1
):
    """
    Main finetuning function with DDP and mixed precision support.
    
    Args:
        use_amp: Enable automatic mixed precision (float16)
        accumulation_steps: Gradient accumulation steps (effective batch = batch_size * accumulation_steps)
    """
    
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    
    # Setup
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path(save_dir)
    if is_rank_zero(rank):
        save_dir.mkdir(exist_ok=True)
    
    if is_rank_zero(rank):
        print(f"Finetuning Swin2SR on device: {device}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset}")
        print(f"Epochs: {epochs}")
        print(f"Distributed: {is_distributed}, World Size: {world_size}, Rank: {rank}")
        print(f"Mixed Precision (AMP): {use_amp}")
        print(f"Gradient Accumulation Steps: {accumulation_steps}")
    
    # Initialize experiment
    experiment = LowDoseCTRecon(dataset=dataset)
    geometry = experiment.geometry
    
    # Create normalizer
    normalizer = CTNormalizer(hu_min=-1000, hu_max=2000)
    
    # Create model with physical constraints
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = model_name
    model_params.train_hf_backbone = True  # Enable training
    model_params.do_rescale = False  # Disable rescaling to preserve HU units
    
    model = PhysicallyAwareSwin2SR(geometry, model_params)
    model.to(device)
    
    # Create datasets
    train_dataset, val_dataset = create_finetuning_dataset(experiment, normalizer, device)
    
    # Create dataloaders with DistributedSampler if distributed
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=0
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Setup optimizer - use different learning rates for different parts
    hf_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'model.' in name:  # Hugging Face model parameters
            hf_params.append(param)
        else:  # New parameters (output scaling, etc.)
            new_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': hf_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained
        {'params': new_params, 'lr': learning_rate}  # Higher LR for new parameters
    ])
    
    # Setup mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    # Loss function - MSE for regression
    loss_fn = nn.MSELoss()
    
    # Metrics
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    
    # Training loop
    train_losses = []
    val_losses = []
    val_psnr = []
    val_ssim = []
    
    best_val_loss = float('inf')
    
    if is_rank_zero(rank):
        print("Starting finetuning...")
    
    for epoch in range(epochs):
        # Set sampler epoch for proper shuffling in distributed training
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (fdk_input, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_rank_zero(rank))):
            # Move to device
            fdk_input = fdk_input.to(device)
            target = target.to(device)
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast('cuda'):
                    output = model(fdk_input)
                    loss = loss_fn(output, target) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                output = model(fdk_input)
                loss = loss_fn(output, target) / accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping for stability
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_score = 0.0
        val_ssim_score = 0.0
        
        with torch.no_grad():
            for fdk_input, target in val_loader:
                fdk_input = fdk_input.to(device)
                target = target.to(device)
                
                if use_amp:
                    with autocast('cuda'):
                        output = model(fdk_input)
                        loss = loss_fn(output, target)
                else:
                    output = model(fdk_input)
                    loss = loss_fn(output, target)
                
                val_loss += loss.item()
                
                # Convert back to HU for metrics
                output_hu = normalizer.denormalize(output)
                target_hu = normalizer.denormalize(target)
                
                # Calculate metrics with explicit reduce parameter
                val_psnr_score += psnr_metric(output_hu, target_hu, reduce='mean').item()
                val_ssim_score += ssim_metric(output_hu, target_hu, reduce='mean').item()
        
        val_loss /= len(val_loader)
        val_psnr_score /= len(val_loader)
        val_ssim_score /= len(val_loader)
        
        val_losses.append(val_loss)
        val_psnr.append(val_psnr_score)
        val_ssim.append(val_ssim_score)
        
        # Print progress (only on rank 0)
        if is_rank_zero(rank):
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val PSNR: {val_psnr_score:.2f} dB")
            print(f"  Val SSIM: {val_ssim_score:.4f}")
        
        # Save best model (only on rank 0)
        if is_rank_zero(rank) and val_loss < best_val_loss:
            best_val_loss = val_loss
            # For DDP, use model.module to access the original model
            model_to_save = model.module if is_distributed else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_psnr': val_psnr_score,
                'val_ssim': val_ssim_score,
            }, save_dir / 'best_model.pt')
            if is_rank_zero(rank):
                print(f"  New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Save checkpoint every 10 epochs (only on rank 0)
        if is_rank_zero(rank) and (epoch + 1) % 10 == 0:
            model_to_save = model.module if is_distributed else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final results (only on rank 0)
    if is_rank_zero(rank):
        model_to_save = model.module if is_distributed else model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
        }, save_dir / 'final_model.pt')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'best_val_loss': best_val_loss,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_name': model_name,
            'dataset': dataset,
            'world_size': world_size,
            'use_amp': use_amp
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(val_psnr)
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR')
        
        plt.subplot(1, 3, 3)
        plt.plot(val_ssim)
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nFinetuning completed!")
        print(f"Results saved to: {save_dir}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final validation PSNR: {val_psnr[-1]:.2f} dB")
        print(f"Final validation SSIM: {val_ssim[-1]:.4f}")
    
    # Cleanup distributed training
    if is_distributed:
        cleanup_distributed()
    
    return model, history if is_rank_zero(rank) else None


def test_finetuned_model(model_path, experiment, device=None):
    """
    Test the finetuned model and show results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model_params = Swin2SR.default_parameters()
    model_params.train_hf_backbone = False  # Inference only
    model_params.do_rescale = False
    
    model = PhysicallyAwareSwin2SR(experiment.geometry, model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create normalizer
    normalizer = CTNormalizer(hu_min=-1000, hu_max=2000)
    
    # Test on validation data
    val_dataset = experiment.get_validation_dataset()
    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print("Testing finetuned model...")
    
    with torch.no_grad():
        for i, (sinogram, ground_truth) in enumerate(test_loader):
            if i >= 3:  # Test on 3 samples
                break
                
            sinogram = sinogram.to(device)
            ground_truth = ground_truth.to(device)
            
            # FDK reconstruction
            from LION.classical_algorithms.fdk import fdk
            fdk_recon = fdk(sinogram, experiment.geometry, clip=True)
            if fdk_recon.dim() == 3:
                fdk_recon = fdk_recon.unsqueeze(1)
            
            # Normalize for model
            fdk_normalized = normalizer.normalize(fdk_recon)
            
            # Model prediction
            output_normalized = model(fdk_normalized)
            
            # Convert back to HU
            fdk_hu = fdk_recon.squeeze().cpu().numpy()
            enhanced_hu = normalizer.denormalize(output_normalized).squeeze().cpu().numpy()
            ground_truth_hu = ground_truth.squeeze().cpu().numpy()
            
            # Print value ranges
            print(f"\nSample {i+1}:")
            print(f"  Ground Truth HU range: {ground_truth_hu.min():.1f} to {ground_truth_hu.max():.1f}")
            print(f"  FDK HU range: {fdk_hu.min():.1f} to {fdk_hu.max():.1f}")
            print(f"  Enhanced HU range: {enhanced_hu.min():.1f} to {enhanced_hu.max():.1f}")
            
            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(ground_truth_hu, cmap='gray', vmin=-1000, vmax=2000)
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(fdk_hu, cmap='gray', vmin=-1000, vmax=2000)
            axes[1].set_title('FDK Reconstruction')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(enhanced_hu, cmap='gray', vmin=-1000, vmax=2000)
            axes[2].set_title('Finetuned Swin2SR')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Example usage
    print("Swin2SR Finetuning for Physically Interpretable CT Reconstruction")
    print("=" * 70)
    
    # Finetune the model with DDP and mixed precision
    model, history = finetune_swin2sr(
        model_name="caidas/swin2SR-classical-sr-x2-64",
        dataset="LIDC-IDRI",
        epochs=50,  # Start with fewer epochs for testing
        batch_size=1,  # Per-GPU batch size
        learning_rate=1e-5,
        save_dir="finetuned_swin2sr_results",
        use_amp=True,  # Enable mixed precision
        accumulation_steps=2  # Effective batch = 1 * 2 = 2
    )
    
    # Test the finetuned model (only on rank 0)
    if history is not None:
        experiment = LowDoseCTRecon(dataset="LIDC-IDRI")
        test_finetuned_model(
            "finetuned_swin2sr_results/best_model.pt",
            experiment
        )
