#!/usr/bin/env python3
"""
Inference script for finetuned Swin2SR model.
Compares FDK reconstruction with HF-enhanced reconstruction.
Produces visualization similar to the training comparison plots.
"""

import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

# LION imports
from LION.models.CNNs.huggingface import Swin2SR
from LION.experiments.ct_experiments import LowDoseCTRecon
from LION.metrics.psnr import PSNR
from LION.metrics.ssim import SSIM


class CTNormalizer:
    """Custom normalizer that preserves CT units while enabling stable training"""
    
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


class PhysicallyAwareSwin2SR(nn.Module):
    """
    Swin2SR with modifications for physically interpretable outputs
    """
    
    def __init__(self, geometry, model_parameters=None):
        super().__init__()
        self.swin2sr = Swin2SR(geometry, model_parameters)
        
        # Add output scaling layer to ensure proper HU range
        self.output_scaling = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        """Forward pass with physical constraints"""
        base_output = self.swin2sr(x)
        scaled_output = base_output * self.output_scaling + self.output_bias
        return scaled_output


def load_finetuned_model(checkpoint_path, geometry, device):
    """Load the finetuned model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model_params = Swin2SR.default_parameters()
    model_params.hf_model_name = "caidas/swin2SR-classical-sr-x2-64"
    model_params.train_hf_backbone = False  # Inference only
    model_params.do_rescale = False
    
    model = PhysicallyAwareSwin2SR(geometry, model_params)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Handle key mismatch: saved keys may be prefixed with "swin2sr.model."
    # but model expects "swin2sr." (DDP vs non-DDP mismatch)
    corrected_state_dict = {}
    for key, value in state_dict.items():
        # If key starts with "swin2sr.model.", it came from DDP wrapping
        # Remove the extra "model." prefix for non-DDP loading
        if key.startswith("swin2sr.model."):
            # Change "swin2sr.model.swin2sr..." to "swin2sr.swin2sr..."
            new_key = key.replace("swin2sr.model.", "swin2sr.", 1)
            corrected_state_dict[new_key] = value
        elif key.startswith("model."):
            # Also handle "model.swin2sr..." -> "swin2sr..."
            new_key = key[6:]  # Remove "model." prefix
            corrected_state_dict[new_key] = value
        else:
            corrected_state_dict[key] = value
    
    model.load_state_dict(corrected_state_dict, strict=False)
    model.eval()
    
    return model


def run_inference(
    model_path,
    output_dir="inference_results",
    num_samples=5,
    dataset="LIDC-IDRI",
    device=None,
    save_metrics=True
):
    """
    Run inference on finetuned model and produce comparisons.
    
    Args:
        model_path: Path to checkpoint (.pt file)
        output_dir: Where to save results
        num_samples: Number of test samples to process
        dataset: Which dataset to use
        device: GPU/CPU device
        save_metrics: Whether to save PSNR/SSIM metrics to file
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Initialize experiment and geometry
    experiment = LowDoseCTRecon(dataset=dataset)
    geometry = experiment.geometry
    
    # Load model
    model = load_finetuned_model(model_path, geometry, device)
    
    # Create normalizer
    normalizer = CTNormalizer(hu_min=-1000, hu_max=2000)
    
    # Get validation dataset
    val_dataset = experiment.get_validation_dataset()
    
    # Metrics
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    
    metrics_list = []
    
    print(f"\nRunning inference on {num_samples} samples...")
    
    with torch.no_grad():
        for sample_idx in tqdm(range(min(num_samples, len(val_dataset))), desc="Inference"):
            sinogram, ground_truth = val_dataset[sample_idx]
            
            # Move to device
            sinogram = sinogram.to(device)
            ground_truth = ground_truth.to(device)
            
            # FDK reconstruction
            from LION.classical_algorithms.fdk import fdk
            fdk_recon = fdk(sinogram, geometry, clip=True)
            
            # Ensure correct shape: (H, W) -> (1, H, W) -> (1, 1, H, W)
            if fdk_recon.dim() == 2:
                fdk_recon = fdk_recon.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
            elif fdk_recon.dim() == 3:
                fdk_recon = fdk_recon.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
            elif fdk_recon.dim() == 4 and fdk_recon.size(0) != 1:
                fdk_recon = fdk_recon.unsqueeze(0)  # Add batch dimension if missing
            
            # Normalize for model
            fdk_normalized = normalizer.normalize(fdk_recon)
            
            # Model prediction (HF enhancement)
            output_normalized = model(fdk_normalized)
            
            # Convert back to HU
            fdk_hu = fdk_recon.squeeze().cpu().numpy()
            enhanced_hu = normalizer.denormalize(output_normalized).squeeze().cpu().numpy()
            ground_truth_hu = ground_truth.squeeze().cpu().numpy()
            
            # Use fixed range for visualization
            vmin, vmax = 0, 2.2
            
            # Ensure ground_truth has batch dimension for metrics
            if ground_truth.dim() == 3:
                ground_truth = ground_truth.unsqueeze(0)
            
            # Compute metrics
            fdk_psnr_tensor = psnr_metric(fdk_recon, ground_truth)
            fdk_psnr = fdk_psnr_tensor.item() if torch.is_tensor(fdk_psnr_tensor) else fdk_psnr_tensor
            
            fdk_ssim_tensor = ssim_metric(fdk_recon, ground_truth)
            fdk_ssim = fdk_ssim_tensor.item() if torch.is_tensor(fdk_ssim_tensor) else fdk_ssim_tensor
            
            enhanced_psnr_tensor = psnr_metric(output_normalized, ground_truth)
            enhanced_psnr = enhanced_psnr_tensor.item() if torch.is_tensor(enhanced_psnr_tensor) else enhanced_psnr_tensor
            
            enhanced_ssim_tensor = ssim_metric(output_normalized, ground_truth)
            enhanced_ssim = enhanced_ssim_tensor.item() if torch.is_tensor(enhanced_ssim_tensor) else enhanced_ssim_tensor
            
            metrics_list.append({
                'sample': sample_idx,
                'fdk_psnr': fdk_psnr.item() if isinstance(fdk_psnr, torch.Tensor) else fdk_psnr,
                'fdk_ssim': fdk_ssim.item() if isinstance(fdk_ssim, torch.Tensor) else fdk_ssim,
                'enhanced_psnr': enhanced_psnr.item() if isinstance(enhanced_psnr, torch.Tensor) else enhanced_psnr,
                'enhanced_ssim': enhanced_ssim.item() if isinstance(enhanced_ssim, torch.Tensor) else enhanced_ssim,
                'psnr_improvement': (enhanced_psnr - fdk_psnr).item() if isinstance(enhanced_psnr, torch.Tensor) else (enhanced_psnr - fdk_psnr),
                'ssim_improvement': (enhanced_ssim - fdk_ssim).item() if isinstance(enhanced_ssim, torch.Tensor) else (enhanced_ssim - fdk_ssim),
            })
            
            # Create comparison figure (6-panel layout)
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Sample {sample_idx + 1}: FDK vs Swin2SR Enhancement', fontsize=16, fontweight='bold')
            
            # Row 1: Reconstructions
            # Ground Truth
            im1 = axes[0, 0].imshow(ground_truth_hu, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f'Ground Truth', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], label='HU')
            
            # FDK Reconstruction
            im2 = axes[0, 1].imshow(fdk_hu, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(
                f'FDK Reconstruction\nPSNR: {fdk_psnr:.2f} dB, SSIM: {fdk_ssim:.4f}',
                fontsize=12, fontweight='bold'
            )
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], label='HU')
            
            # HF Enhanced
            im3 = axes[0, 2].imshow(enhanced_hu, cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, 2].set_title(
                f'HF Enhanced (Swin2SR)\nPSNR: {enhanced_psnr:.2f} dB, SSIM: {enhanced_ssim:.4f}',
                fontsize=12, fontweight='bold', color='green'
            )
            axes[0, 2].axis('off')
            plt.colorbar(im3, ax=axes[0, 2], label='HU')
            
            # Row 2: Error Maps
            # FDK Error Map
            fdk_error = np.abs(fdk_hu - ground_truth_hu)
            im4 = axes[1, 0].imshow(fdk_error, cmap='hot', vmin=0, vmax=500)
            axes[1, 0].set_title(
                f'FDK Error Map\nMean Absolute Error: {fdk_error.mean():.2f} HU',
                fontsize=12
            )
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0], label='|Error| (HU)')
            
            # Enhanced Error Map
            enhanced_error = np.abs(enhanced_hu - ground_truth_hu)
            im5 = axes[1, 1].imshow(enhanced_error, cmap='hot', vmin=0, vmax=500)
            axes[1, 1].set_title(
                f'Enhanced Error Map\nMean Absolute Error: {enhanced_error.mean():.2f} HU',
                fontsize=12, color='green'
            )
            axes[1, 1].axis('off')
            plt.colorbar(im5, ax=axes[1, 1], label='|Error| (HU)')
            
            # Improvement Map (negative = improvement)
            improvement = fdk_error - enhanced_error
            im6 = axes[1, 2].imshow(improvement, cmap='RdBu_r', vmin=-200, vmax=200)
            axes[1, 2].set_title(
                f'Improvement (FDK Error - Enhanced Error)\nMean: {improvement.mean():.2f} HU',
                fontsize=12
            )
            axes[1, 2].axis('off')
            cbar = plt.colorbar(im6, ax=axes[1, 2], label='Error Reduction (HU)')
            
            plt.tight_layout()
            
            # Save figure
            save_path = output_dir / f'sample_{sample_idx:03d}_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()
            
            # Print sample metrics
            print(f"\nSample {sample_idx + 1}:")
            print(f"  Ground Truth HU range: {ground_truth_hu.min():.1f} to {ground_truth_hu.max():.1f}")
            print(f"  FDK HU range: {fdk_hu.min():.1f} to {fdk_hu.max():.1f}")
            print(f"  Enhanced HU range: {enhanced_hu.min():.1f} to {enhanced_hu.max():.1f}")
            print(f"  FDK → PSNR: {fdk_psnr:.2f} dB, SSIM: {fdk_ssim:.4f}")
            print(f"  Enhanced → PSNR: {enhanced_psnr:.2f} dB, SSIM: {enhanced_ssim:.4f}")
            print(f"  Improvement → PSNR: {metrics_list[-1]['psnr_improvement']:+.2f} dB, SSIM: {metrics_list[-1]['ssim_improvement']:+.4f}")
    
    # Save metrics summary
    if save_metrics and metrics_list:
        import json
        metrics_file = output_dir / 'metrics_summary.json'
        
        # Compute aggregate stats
        fdk_psnr_list = [m['fdk_psnr'] for m in metrics_list]
        fdk_ssim_list = [m['fdk_ssim'] for m in metrics_list]
        enhanced_psnr_list = [m['enhanced_psnr'] for m in metrics_list]
        enhanced_ssim_list = [m['enhanced_ssim'] for m in metrics_list]
        psnr_imp_list = [m['psnr_improvement'] for m in metrics_list]
        ssim_imp_list = [m['ssim_improvement'] for m in metrics_list]
        
        summary = {
            'num_samples': len(metrics_list),
            'fdk': {
                'mean_psnr': float(np.mean(fdk_psnr_list)),
                'std_psnr': float(np.std(fdk_psnr_list)),
                'mean_ssim': float(np.mean(fdk_ssim_list)),
                'std_ssim': float(np.std(fdk_ssim_list)),
            },
            'enhanced': {
                'mean_psnr': float(np.mean(enhanced_psnr_list)),
                'std_psnr': float(np.std(enhanced_psnr_list)),
                'mean_ssim': float(np.mean(enhanced_ssim_list)),
                'std_ssim': float(np.std(enhanced_ssim_list)),
            },
            'improvement': {
                'mean_psnr': float(np.mean(psnr_imp_list)),
                'std_psnr': float(np.std(psnr_imp_list)),
                'mean_ssim': float(np.mean(ssim_imp_list)),
                'std_ssim': float(np.std(ssim_imp_list)),
            },
            'samples': metrics_list
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*70}")
        print("METRICS SUMMARY")
        print(f"{'='*70}")
        print(f"\nFDK Reconstruction:")
        print(f"  Mean PSNR: {summary['fdk']['mean_psnr']:.2f} ± {summary['fdk']['std_psnr']:.2f} dB")
        print(f"  Mean SSIM: {summary['fdk']['mean_ssim']:.4f} ± {summary['fdk']['std_ssim']:.4f}")
        
        print(f"\nHF Enhanced (Swin2SR):")
        print(f"  Mean PSNR: {summary['enhanced']['mean_psnr']:.2f} ± {summary['enhanced']['std_psnr']:.2f} dB")
        print(f"  Mean SSIM: {summary['enhanced']['mean_ssim']:.4f} ± {summary['enhanced']['std_ssim']:.4f}")
        
        print(f"\nImprovement (Enhanced - FDK):")
        print(f"  Mean PSNR: {summary['improvement']['mean_psnr']:+.2f} ± {summary['improvement']['std_psnr']:.2f} dB")
        print(f"  Mean SSIM: {summary['improvement']['mean_ssim']:+.4f} ± {summary['improvement']['std_ssim']:.4f}")
        
        print(f"\nMetrics saved to: {metrics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for finetuned Swin2SR model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="finetuned_swin2sr_results/best_model.pt",
        help="Path to finetuned model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of test samples to process"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LIDC-IDRI",
        help="Dataset to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    run_inference(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        dataset=args.dataset,
        device=device
    )
