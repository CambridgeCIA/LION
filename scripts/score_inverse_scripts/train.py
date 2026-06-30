"""
Distributed training pipeline for NCSN++ score-matching networks.

Processing Flow:
  1. Initializes PyTorch DistributedDataParallel (DDP) for multi-GPU training.
  2. Loads slices from the LIDC-IDRI dataset on CPU.
  3. Truncates training data size according to user configuration (--data_prop).
  4. Optimizes model weights using the Denoising Score Matching loss function.
  5. Periodically updates Exponential Moving Average (EMA) shadow weights and saves training checkpoints.

Author: Tianzhen Peng

References
----------
.. [Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR. https://arxiv.org/abs/2011.13456
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from LION.data_loaders.LIDC_IDRI import LIDC_IDRI
from LION.CTtools.ct_utils import from_HU_to_normal
from LION.models.score_inverse.ema import EMA
from LION.models.score_inverse.ncsnpp import NCSNpp, get_score_fn
from LION.models.score_inverse.sde import VESDE
from LION.models.score_inverse.loss import SMLoss
from LION.models.score_inverse.utils import set_global_seed

CONFIG = {
    "global_batch_size": 16,
    "loader_num_workers": 8,
    "snapshot_freq": 5,
    "lr": 2e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.0,
    "warmup_steps": 5000,
    "grad_clip": 1.0,
    "ema_rate": 0.9999,
    "sigma_min": 0.01,
    "sigma_max": 220.0,
    "model_kwargs": {
        "image_resolution": 512,
        "num_channels": 1,
        "nf": 16,
        "ch_mult": (1, 2, 4, 8, 16, 32, 32),
        "num_res_blocks": 1,
        "attn_resolutions": (16,),
        "dropout": 0.0,
        "fourier_scale": 16.0
    }
}

def seed_worker(worker_id):
    """Reproducibility function for dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader(config, data_prop, seed, local_rank, world_size, is_distributed):
    assert config['global_batch_size'] % world_size == 0, "Global batch size must be divisible by world size"
    batch_size_per_gpu = config['global_batch_size'] // world_size
    
    # Load training dataset
    params = LIDC_IDRI.default_parameters(task="image_only")
    params.max_num_slices_per_patient = -1
    # Keep dataset on CPU to avoid CUDA re-init in forked DataLoader workers.
    params.device = torch.device("cpu")
    dataset = LIDC_IDRI(mode="train", parameters=params)

    # Subset the dataset if data_prop < 1.0
    if data_prop < 1.0:
        subset_size = int(len(dataset) * data_prop)
        subset_generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=subset_generator)[:subset_size].tolist()
        dataset = Subset(dataset, indices)
        if local_rank == 0:
            print(f"Subsetting Active: {subset_size} images ({data_prop*100}%).")

    if is_distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True, seed=seed, drop_last=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config['loader_num_workers'],
        pin_memory=True,
        worker_init_fn=seed_worker,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return loader, sampler

def load_checkpoint(path, model, ema, optimizer, device):
    if path is None:
        return None
    
    # Normalize device for torch.load's map_location. Accept either an int (GPU rank), a string like 'cpu' or 'cuda:0', or a torch.device.
    map_location = f'cuda:{device}' if isinstance(device, int) else device
    ckpt = torch.load(path, map_location=map_location)
    
    model.load_state_dict(ckpt['model_state_dict'])
    if ema is not None:
        ema.load_state_dict(ckpt['ema_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt

def save_checkpoint(save_dir, step, epoch, model, ema, optimizer):
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth")
    ckpt = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(ckpt, checkpoint_path)
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_prop', type=float, default=1.0, help="Proportion of training data")
    parser.add_argument('--epochs', type=int, default=200, help="Total epochs")
    parser.add_argument('--seed', type=int, default=0, help="Global random seed")
    parser.add_argument('--ema_rate', type=float, default=CONFIG['ema_rate'], help="EMA decay rate")
    parser.add_argument('--save_path', type=str, default='checkpoints', help="Directory to store model checkpoints")
    parser.add_argument('--resume_checkpoint', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--snapshot_freq', type=int, default=CONFIG['snapshot_freq'], help="Frequency to store model checkpoints (in epoch).")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # Dynamic DDP Initialization
    is_distributed = "LOCAL_RANK" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    set_global_seed(args.seed)

    train_loader, train_sampler = get_dataloader(
        CONFIG, args.data_prop, args.seed, local_rank, world_size, is_distributed
    )

    sde = VESDE(sigma_min=CONFIG['sigma_min'], sigma_max=CONFIG['sigma_max'])

    model = NCSNpp(**CONFIG['model_kwargs']).to(local_rank)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    ema = EMA(unwrapped_model, args.ema_rate) if local_rank == 0 else None

    score_fn = get_score_fn(model, sde)
    loss_fn = SMLoss(score_fn, sde, eps=1e-5)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG['lr'], betas=(CONFIG['beta1'], CONFIG['beta2']), 
        eps=CONFIG['eps'], weight_decay=CONFIG['weight_decay']
    )

    ckpt = None
    if args.resume_checkpoint is not None:
        ckpt = load_checkpoint(args.resume_checkpoint, unwrapped_model, ema, optimizer, device=local_rank)

    steps_per_epoch = len(train_loader)
    last_epoch = int(ckpt.get('epoch', -1)) if ckpt is not None else -1
    global_step = (last_epoch + 1) * steps_per_epoch - 1
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(step / CONFIG['warmup_steps'], 1.0), last_epoch=global_step,
    )

    if local_rank == 0:
        print(f"Starting NCSN++ Training -- GPUs: {world_size} | Global Batch: {CONFIG['global_batch_size']} | Epochs: {args.epochs} | Seed: {args.seed}")

    for epoch in range(last_epoch + 1, args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        losses = []
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            x = from_HU_to_normal(batch.to(local_rank, non_blocking=True))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(x)

            # Gather losses from all GPUs to compute step-level global batch loss
            if is_distributed:
                loss_list = [torch.zeros_like(loss) for _ in range(world_size)] if local_rank == 0 else None
                dist.gather(loss.detach(), loss_list, dst=0)
                
                if local_rank == 0:
                    # Average loss across the full global batch
                    step_loss = torch.stack(loss_list).mean()
                    losses.append(step_loss)
            else:
                losses.append(loss.detach())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            optimizer.step()
            scheduler.step()
            if local_rank == 0:
                ema.update()
            
            global_step += 1
            if local_rank == 0 and global_step % 100 == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
        if local_rank == 0 and ((epoch + 1) % args.snapshot_freq == 0 or (epoch + 1) == args.epochs):
            checkpoint_path = save_checkpoint(args.save_path, global_step, epoch, unwrapped_model, ema, optimizer)
            print(f"Epoch {epoch} | Saved checkpoint: {checkpoint_path}")

        # step-level global batch loss
        if local_rank == 0:
            losses = torch.stack(losses).cpu().numpy()  # shape: (num_steps,)
            print(f"Epoch {epoch}/{args.epochs} | Epoch Loss Mean: {np.mean(losses):.4f}, Std: {np.std(losses):.4f}, Min: {np.min(losses):.4f}, Max: {np.max(losses):.4f}")
            if args.save_path is not None:
                np.save(os.path.join(args.save_path, f"loss_epoch_{epoch}.npy"), losses.astype(np.float32))

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()