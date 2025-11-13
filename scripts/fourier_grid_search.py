#!/usr/bin/env python
"""
CryoGEN Fourier Masking Hyperparameter Grid Search

This script performs a comprehensive grid search over CryoGEN hyperparameters
using Fourier domain masking on a single cryo-EM image. It explores combinations
of mask probability, zeta_scale, beta, and number of masks to identify optimal
reconstruction parameters.

Key Parameters:
- mask_prob: Probability of sampling each Fourier coefficient (0.1 to 0.5)
- zeta_scale: Step size for measurement consistency gradient
- beta: Nesterov momentum coefficient
- num_masks: Number of random Fourier masks to use

Usage:
    python scripts/fourier_grid_search.py
"""

import os
import sys
import torch
import pandas as pd
from itertools import product
from pathlib import Path

# Add CryoGEN to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CryoGEN import CryoGEN


def grid_search_fourier():
    """
    Perform grid search over Fourier masking hyperparameters.
    """

    # =====================================================================
    # GRID SEARCH CONFIGURATION
    # =====================================================================

    # Model and data configuration
    MODEL_PATH = "anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128"
    DATA_PATH = "data/sample_empiar10076.pt"
    IMAGE_ID = 0  # Single image for grid search
    BLOCK_SIZE = 1  # No downsampling (full resolution)
    MASK_TYPE = "random_fourier"

    # Hyperparameter grid
    PARAM_GRID = {
        'mask_prob': [0.1,0.2,0.3,0.4,0.5],  # Fourier coefficient sampling probability
        'num_masks': [1],  # Number of random Fourier masks
        'zeta_scale': [0.1, 0.5, 1.0, 5.0, 10.0],  # Measurement consistency step size
        'beta': [0.9],  # Nesterov momentum
    }

    # Fixed parameters
    NUM_TIMESTEPS = 1000  # Full DDPM schedule
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output configuration
    BASE_RESULT_DIR = "results/fourier_grid_search"
    os.makedirs(BASE_RESULT_DIR, exist_ok=True)

    # =====================================================================
    # INITIALIZE GRID SEARCH
    # =====================================================================

    print("=" * 70)
    print("CryoGEN FOURIER MASKING GRID SEARCH")
    print("=" * 70)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Image ID: {IMAGE_ID}")
    print(f"Block Size: {BLOCK_SIZE}")
    print(f"Mask Type: {MASK_TYPE}")
    print(f"Device: {DEVICE}")
    print(f"\nParameter Grid:")
    for param, values in PARAM_GRID.items():
        print(f"  {param}: {values}")

    # Calculate total experiments
    total_experiments = 1
    for values in PARAM_GRID.values():
        total_experiments *= len(values)

    print(f"\nTotal experiments: {total_experiments}")
    print("=" * 70)
    print()

    # =====================================================================
    # RUN GRID SEARCH
    # =====================================================================

    results = []
    experiment_num = 0

    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    for params in product(*param_values):
        experiment_num += 1

        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))

        # Extract parameters
        mask_prob = param_dict['mask_prob']
        num_masks = param_dict['num_masks']
        zeta_scale = param_dict['zeta_scale']
        beta = param_dict['beta']

        print("-" * 70)
        print(f"Experiment {experiment_num}/{total_experiments}")
        print(f"Parameters: mask_prob={mask_prob}, num_masks={num_masks}, "
              f"zeta_scale={zeta_scale}, beta={beta}")
        print("-" * 70)

        # Create experiment-specific result directory
        exp_name = f"maskprob{mask_prob}_masks{num_masks}_zeta{zeta_scale}_beta{beta}"
        result_dir = os.path.join(BASE_RESULT_DIR, exp_name)
        os.makedirs(result_dir, exist_ok=True)

        try:
            # Initialize CryoGEN (use_config=False for manual parameter control)
            cryogen = CryoGEN(
                model_path=MODEL_PATH,
                block_size=BLOCK_SIZE,
                device=DEVICE,
                result_dir=result_dir,  # Set result directory in constructor
                use_config=False,  # Manual hyperparameter control
                verbose=False  # Disable verbose to reduce disk usage
            )

            # Run reconstruction
            reconstructed, original, metrics = cryogen.reconstruct_from_cryoem(
                file_path=DATA_PATH,
                image_ids=[IMAGE_ID],
                num_masks=num_masks,
                mask_type=MASK_TYPE,
                mask_prob=mask_prob,
                zeta_scale=zeta_scale,
                beta=beta,
                beta_min=0.1,  # Standard starting value
                zeta_min=1e-2,  # Standard starting value
                num_timesteps=NUM_TIMESTEPS
            )

            # Extract metrics for the single image
            img_metrics = metrics[0]

            # Store results
            result_entry = {
                'experiment': experiment_num,
                'mask_prob': mask_prob,
                'num_masks': num_masks,
                'zeta_scale': zeta_scale,
                'beta': beta,
                'psnr': img_metrics['psnr'],
                'ssim': img_metrics['ssim'],
                'lpips': img_metrics['lpips'],
                'mse': img_metrics['mse'],
                'mae': img_metrics['mae'],
                'measurement_mse': img_metrics['measurement_mse'],
                'result_dir': result_dir,
                'status': 'success'
            }

            print(f"✓ PSNR: {img_metrics['psnr']:.2f} dB | "
                  f"SSIM: {img_metrics['ssim']:.4f} | "
                  f"LPIPS: {img_metrics['lpips']:.4f}")

        except Exception as e:
            print(f"✗ Experiment failed: {str(e)}")
            result_entry = {
                'experiment': experiment_num,
                'mask_prob': mask_prob,
                'num_masks': num_masks,
                'zeta_scale': zeta_scale,
                'beta': beta,
                'psnr': None,
                'ssim': None,
                'lpips': None,
                'mse': None,
                'mae': None,
                'measurement_mse': None,
                'result_dir': result_dir,
                'status': f'failed: {str(e)}'
            }

        results.append(result_entry)

        # Save intermediate results after each experiment
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(BASE_RESULT_DIR, 'grid_search_results.csv'), index=False)

        print()

    # =====================================================================
    # ANALYZE RESULTS
    # =====================================================================

    print("=" * 70)
    print("GRID SEARCH COMPLETE - ANALYZING RESULTS")
    print("=" * 70)
    print()

    # Load final results
    df = pd.DataFrame(results)

    # Filter successful experiments
    df_success = df[df['status'] == 'success'].copy()

    if len(df_success) == 0:
        print("⚠ No successful experiments!")
        return

    # Find best parameters for each metric
    print("BEST PARAMETERS BY METRIC:")
    print("-" * 70)

    # Best PSNR
    best_psnr = df_success.loc[df_success['psnr'].idxmax()]
    print(f"\nBest PSNR: {best_psnr['psnr']:.2f} dB")
    print(f"  mask_prob={best_psnr['mask_prob']}, num_masks={best_psnr['num_masks']}, "
          f"zeta_scale={best_psnr['zeta_scale']}, beta={best_psnr['beta']}")

    # Best SSIM
    best_ssim = df_success.loc[df_success['ssim'].idxmax()]
    print(f"\nBest SSIM: {best_ssim['ssim']:.4f}")
    print(f"  mask_prob={best_ssim['mask_prob']}, num_masks={best_ssim['num_masks']}, "
          f"zeta_scale={best_ssim['zeta_scale']}, beta={best_ssim['beta']}")

    # Best LPIPS (lower is better)
    best_lpips = df_success.loc[df_success['lpips'].idxmin()]
    print(f"\nBest LPIPS: {best_lpips['lpips']:.4f}")
    print(f"  mask_prob={best_lpips['mask_prob']}, num_masks={best_lpips['num_masks']}, "
          f"zeta_scale={best_lpips['zeta_scale']}, beta={best_lpips['beta']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (Successful Experiments)")
    print("=" * 70)
    print(f"\nTotal experiments: {len(df)}")
    print(f"Successful: {len(df_success)}")
    print(f"Failed: {len(df) - len(df_success)}")

    print(f"\nPSNR: {df_success['psnr'].mean():.2f} ± {df_success['psnr'].std():.2f} dB "
          f"(range: {df_success['psnr'].min():.2f} - {df_success['psnr'].max():.2f})")
    print(f"SSIM: {df_success['ssim'].mean():.4f} ± {df_success['ssim'].std():.4f} "
          f"(range: {df_success['ssim'].min():.4f} - {df_success['ssim'].max():.4f})")
    print(f"LPIPS: {df_success['lpips'].mean():.4f} ± {df_success['lpips'].std():.4f} "
          f"(range: {df_success['lpips'].min():.4f} - {df_success['lpips'].max():.4f})")

    # Analyze parameter importance (correlation with PSNR)
    print("\n" + "=" * 70)
    print("PARAMETER CORRELATION WITH PSNR")
    print("=" * 70)
    for param in param_names:
        corr = df_success[param].corr(df_success['psnr'])
        print(f"{param}: {corr:+.3f}")

    # Top 10 configurations
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS (by PSNR)")
    print("=" * 70)
    top10 = df_success.nlargest(10, 'psnr')
    print(top10[['mask_prob', 'num_masks', 'zeta_scale', 'beta', 'psnr', 'ssim', 'lpips']].to_string(index=False))

    # Save final results
    final_csv = os.path.join(BASE_RESULT_DIR, 'grid_search_results_final.csv')
    df.to_csv(final_csv, index=False)

    # Save summary
    summary_file = os.path.join(BASE_RESULT_DIR, 'grid_search_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CRYOGEN FOURIER MASKING GRID SEARCH SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Successful: {len(df_success)}\n")
        f.write(f"Failed: {len(df) - len(df_success)}\n\n")
        f.write("BEST PARAMETERS:\n")
        f.write(f"\nBest PSNR ({best_psnr['psnr']:.2f} dB):\n")
        f.write(f"  mask_prob={best_psnr['mask_prob']}, num_masks={best_psnr['num_masks']}, "
                f"zeta_scale={best_psnr['zeta_scale']}, beta={best_psnr['beta']}\n")
        f.write(f"\nBest SSIM ({best_ssim['ssim']:.4f}):\n")
        f.write(f"  mask_prob={best_ssim['mask_prob']}, num_masks={best_ssim['num_masks']}, "
                f"zeta_scale={best_ssim['zeta_scale']}, beta={best_ssim['beta']}\n")
        f.write(f"\nBest LPIPS ({best_lpips['lpips']:.4f}):\n")
        f.write(f"  mask_prob={best_lpips['mask_prob']}, num_masks={best_lpips['num_masks']}, "
                f"zeta_scale={best_lpips['zeta_scale']}, beta={best_lpips['beta']}\n")

    print(f"\n\nResults saved to: {BASE_RESULT_DIR}")
    print(f"  - CSV: grid_search_results_final.csv")
    print(f"  - Summary: grid_search_summary.txt")
    print()


if __name__ == "__main__":
    grid_search_fourier()
