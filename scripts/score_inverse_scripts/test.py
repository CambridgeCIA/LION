"""
Downstream evaluation runner for sparse-view CT reconstructions.

Processing Flow:
  1. Sets up the forward Radon transform (SIRT or FST).
  2. Loads input test sinograms and instantiates the NCSN++ score model.
  3. PC sampling with data-consistency hijacking.
  5. Saves output reconstructions along with execution metrics (time, memory).

Author: Tianzhen Peng

References
----------
.. [Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR. https://arxiv.org/abs/2011.13456

.. [Song2022] Song, Y., Shen, L., Xing, L., & Ermon, S. (2022). "Solving Inverse Problems in Medical Imaging with Score-Based Generative Models." ICLR. https://arxiv.org/abs/2206.00364
"""

import argparse
import os
from time import time
import resource
import numpy as np
import torch

from LION.CTtools.ct_utils import make_operator
from LION.models.score_inverse.ema import EMA
from LION.models.score_inverse.ncsnpp import NCSNpp, get_score_fn
from LION.models.score_inverse.sampling import pc_sampler, pc_sampler_new, ReverseDiffusionVE, LangevinDynamics, get_hijack, get_hijack_new
from LION.models.score_inverse.utils import load_checkpoint_eval, set_global_seed, batch_apply
from LION.models.score_inverse.fst import FSTRadon
from LION.models.score_inverse.sirt_adj import SIRTAdj

from configs import sparse_geometry, sparse_geometry_fan, data_dir, sde

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, nargs='+', default=['/home/tp534/rds/hpc-work/trained_models/ncsnpp/checkpoint_epoch_199_step_2376399.pth'], help='Path to the checkpoint file(s)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run validation on (GPU is assumed and required)')
    parser.add_argument('--lb', type=float, nargs='+', default=[0.841], help='Parameter lambda for hijacking')
    parser.add_argument('--N', type=int, nargs='+', default=[100], help='Number of predictor steps')
    parser.add_argument('--M', type=int, default=1, help='Number of corrector steps')
    parser.add_argument('--snr', type=float, default=0.246, help='Corrector SNR')
    parser.add_argument('--use_old_sampler', action='store_true', help='If set, use the old pc_sampler instead of pc_sampler_new')
    parser.add_argument('--output_dir', type=str, default=data_dir, help='Directory to save the output reconstructions')
    parser.add_argument('--output_name', type=str, default='score_inv', help='Reconstruction name for output saving')
    parser.add_argument('--compile', action='store_true', help='If set, compile the model using torch.compile')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 precision for validation (only effective on CUDA devices)')
    parser.add_argument('--sinogram_path', type=str, default=None, help='Path to precalculated .npy sinogram file')
    parser.add_argument('--pseudo_inv_mode', type=str, default='sirt', choices=['fst', 'sirt'], help='Reconstruction operator to use for pseudo-inverse')
    parser.add_argument('--geometry', type=str, default='parallel', choices=['parallel', 'fan'], help='CT geometry type')
    parser.add_argument('--clean_hijack', action='store_true', help='If set, use clean hijacking (no noise injection to the sinogram in hijack function)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Strictly enforce and assume CUDA GPU availability
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required and assumed for this script, but was not found or selected.")
    
    print('Test parameters:')
    for arg in vars(args):
        print(f'  {arg}: {getattr(args, arg)}')

    image_resolution = 512
    
    # Select geometry and validate options
    if args.geometry == 'fan':
        if args.pseudo_inv_mode == 'fst':
            raise ValueError("Fourier Slice Theorem (FST) mode is not supported for fan-beam geometry because the Fourier Slice Theorem only holds for parallel projections.")
        geo = sparse_geometry_fan
    else:
        geo = sparse_geometry

    raw_sparse_op = make_operator(geo)
    sparse_op = batch_apply(raw_sparse_op)
    
    if args.pseudo_inv_mode == 'sirt':
        pseudo_inv = SIRTAdj(raw_sparse_op, device=device)
    elif args.pseudo_inv_mode == 'fst':
        fst_op = FSTRadon(geo=geo, expansion=6, device=device)
    else:
        raise ValueError(f"Unknown pseudo_inv_mode: {args.pseudo_inv_mode}")
        
    print(f"Successfully initialized LION GPU sparse-view projection with recon op: {args.pseudo_inv_mode}.")

    print(f"Loading sinograms from: {args.sinogram_path}")
    sinos = torch.from_numpy(np.load(args.sinogram_path)).to(device)
    args.num_images = sinos.shape[0]
    print(f"Sparse-view sinograms: {sinos.shape}.")
    
    # Initialize model
    model = NCSNpp(
        image_resolution=image_resolution,
        num_channels=1,
        nf=16,
        ch_mult=(1, 2, 4, 8, 16, 32, 32),
        num_res_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        fourier_scale=16.0,
    ).to(device)
    ema = EMA(model)
    
    # Pre-compile the model if requested (done once before parameter sweeps)
    if args.compile:
        print("Compiling model using torch.compile...")
        compile_start_time = time()
        model = torch.compile(model)
        
        # Invoke model with a zero tensor of the desired shape to trigger compilation immediately
        print("Triggering compilation / warming up compiled model with zero inputs...")
        with torch.inference_mode():
            amp_context = torch.amp.autocast('cuda', enabled=(args.bf16 and device.type == 'cuda'), dtype=torch.bfloat16)
            with amp_context:
                dummy_x = torch.zeros((args.num_images, 1, image_resolution, image_resolution), device=device)
                dummy_t = torch.zeros((args.num_images,), device=device)
                _ = model(dummy_x, dummy_t)
            
        print(f"Compilation completed in {time() - compile_start_time:.2f} seconds.")
    
    # Start looping over configuration parameters
    for cp_path in args.checkpoint_path:
        underlying_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        load_checkpoint_eval(cp_path, underlying_model, ema, device)
        print(f"\nLoaded checkpoint weights from: {cp_path}")
        
        # Extract epoch index from filename format e.g. checkpoint_epoch_199_step_2376399.pth
        epoch = int(os.path.basename(cp_path).split("_")[2]) + 1
            
        score_fn = get_score_fn(model, sde)
        
        for N in args.N:
            predictor = ReverseDiffusionVE(sde, score_fn, N=N)
            corrector = LangevinDynamics(score_fn, M=args.M, snr=args.snr)
            
            for lb in args.lb:
                # Check if the output file already exists to enable resuming
                run_name = f"{args.output_name}_{args.pseudo_inv_mode}_epoch{epoch}_N{N}_lb{lb}{'_ch' if args.clean_hijack else ''}_seed{args.seed}"
                output_path = os.path.join(args.output_dir, f"{run_name}.npy")
                if os.path.exists(output_path):
                    print(f"Skipping already completed run: {run_name}")
                    continue
                
                print("=" * 80)
                print(f"Running Reconstruction: Epoch={epoch}, N={N}, Lambda={lb}")
                print("=" * 80)
                
                # Reset random seed for exact reproducibility across runs
                set_global_seed(args.seed)
                
                # Dual-domain data consistency (hijacking)
                if args.pseudo_inv_mode == 'sirt':
                    hijack_fn = get_hijack_new(
                        sde=sde,
                        op=sparse_op,
                        pseudo_inv=pseudo_inv,
                        y=sinos,
                        lb=lb,
                        clean_hijack=args.clean_hijack
                    )
                elif args.pseudo_inv_mode == 'fst':
                    known_kspace = fst_op.sino_to_kspace(sinos)
                    hijack_fn = get_hijack(
                        sde=sde,
                        full_op=fst_op.image_to_kspace,
                        full_op_inv=fst_op.kspace_to_image,
                        y=known_kspace,
                        mask=fst_op.mask,
                        lb=lb,
                        clean_hijack=args.clean_hijack
                    )
                sampler_fn = pc_sampler if args.use_old_sampler else pc_sampler_new
                
                print("Starting diffusion PC sampler...")
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                start_time = time()
                
                amp_context = torch.amp.autocast('cuda', enabled=(args.bf16 and device.type == 'cuda'), dtype=torch.bfloat16)
                with torch.inference_mode():
                    with amp_context:
                        x_noise = sde.sample_noise(shape=(args.num_images, 1, image_resolution, image_resolution)).to(device)
                        recon = sampler_fn(x_noise, predictor, corrector, hijack=hijack_fn, verbose=True, verbose_freq=100) # (B, 1, H, W)
                    recon = recon.float()
                    
                torch.cuda.synchronize()
                end_time = time()
                elapsed_time = end_time - start_time
                
                peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_rss_mb = peak_rss_kb / 1024.0
                peak_memory_bytes = torch.cuda.max_memory_allocated(device)
                peak_memory_mb = peak_memory_bytes / (1024.0 ** 2)
                
                print(f"Reconstruction completed. Output shape: {recon.shape}")
                print(f"Time taken: {elapsed_time:.2f} seconds.")
                print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB.")
                print(f"Peak RSS memory usage: {peak_rss_mb:.2f} MB.")
                
                metrics = {
                    "time_seconds": elapsed_time,
                    "peak_gpu_memory_mb": peak_memory_mb,
                    "peak_rss_memory_mb": peak_rss_mb
                }
                
                
                current_args = vars(args).copy()
                current_args['checkpoint_path'] = cp_path
                current_args['N'] = N
                current_args['lb'] = lb
                current_args['method'] = run_name  # For consistent naming in output files
                
                os.makedirs(args.output_dir, exist_ok=True)
                recon_np = recon.cpu().numpy()
                data_to_save = {
                    "recon": recon_np,
                    "args": current_args,
                    "metrics": metrics
                }
                np.save(output_path, data_to_save)
                print(f"Saved reconstructions to {output_path}")

if __name__ == '__main__':
    main()
