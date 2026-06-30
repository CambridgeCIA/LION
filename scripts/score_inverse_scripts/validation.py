"""
Validation runner for evaluating score network checkpoint loss.

Evaluates trained checkpoints on the LIDC-IDRI validation subset using the stochastic score-matching loss under VESDE noise perturbations. Use seeds to make the validation deterministic.

Author: Tianzhen Peng
"""

import argparse
import os
from LION.CTtools.ct_utils import from_HU_to_normal
from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.models.score_inverse.ema import EMA
from LION.models.score_inverse.ncsnpp import NCSNpp, get_score_fn
from LION.models.score_inverse.sde import VESDE
from LION.models.score_inverse.utils import load_checkpoint_eval, set_global_seed
from LION.models.score_inverse.loss import SMLoss
import torch
from torch.utils.data import DataLoader
from pathlib import Path
# from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for validation (inclusive)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run validation on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image height and width')
    parser.add_argument('--sigma_min', type=float, default=0.01, help='VESDE sigma_min')
    parser.add_argument('--sigma_max', type=float, default=220.0, help='VESDE sigma_max')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision for validation (only effective on CUDA devices)')
    parser.add_argument('--rev_order', action='store_true', help='For reproducibility testing: reverse the order of checkpoints.')
    parser.add_argument('--smloss_seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available, but device was set to cuda')
    
    # Print parameters
    print('Validation parameters:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')

    # Seed global RNGs and enforce deterministic algorithms so dataset and
    # sampling are reproducible across runs.
    set_global_seed(0)
    
    sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    
    # Load validation dataset
    params = LIDC_IDRI.default_parameters(task="image_only")
    params.max_num_slices_per_patient = -1
    params.device = torch.device("cpu")
    dataset = LIDC_IDRI(mode="validation", parameters=params)
    num_workers = min(8, os.cpu_count() or 1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"Loaded validation dataset with {len(dataset)} samples.")
    
    # Load model and checkpoint
    model = NCSNpp(
        image_resolution=args.image_resolution,
        num_channels=1,
        nf=16,
        ch_mult=(1, 2, 4, 8, 16, 32, 32),
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        fourier_scale=16.0,
    )
    ema = EMA(model)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[2]))
    if args.rev_order:
        checkpoint_paths.reverse()
    print(f"Found {len(checkpoint_paths)} checkpoints in {checkpoint_dir}.")
    
    for checkpoint_path in checkpoint_paths:
        if args.start_epoch > 0:
            epoch = int(checkpoint_path.stem.split("_")[2])
            if epoch < args.start_epoch:
                continue
        print(f"Evaluating checkpoint: {checkpoint_path}")
        generator = torch.Generator(device=device)
        generator.manual_seed(args.smloss_seed)  # Seed generator used by SMLoss for deterministic noise
        load_checkpoint_eval(checkpoint_path, model, ema, device)
        model.eval()
        score_fn = get_score_fn(model, sde)
        loss_fn = SMLoss(score_fn, sde, eps=1e-5)
        total_loss = torch.zeros((), device=device)
        total_items = 0
        with torch.inference_mode():
            # for batch in tqdm(loader, desc="Validation", mininterval=2.0, smoothing=0.0):
            for batch in loader:
                x = from_HU_to_normal(batch.to(device, non_blocking=True))
                if args.bf16 and device.type == 'cuda':
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = loss_fn(x, generator=generator)
                else:
                    loss = loss_fn(x, generator=generator)
                batch_size = x.shape[0]
                total_loss += loss.detach() * batch_size
                total_items += batch_size
        total_loss = (total_loss / total_items).item()
        print(f"Checkpoint: {checkpoint_path}, Validation Loss: {total_loss:.4f}")

if __name__ == '__main__':
    main()
