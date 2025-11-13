#!/usr/bin/env python3
"""
Script for reconstructing images using Fourier masking with varying mask probabilities,
distributing the work across multiple GPUs.
"""

import os
import subprocess
import json
import argparse
import math
import numpy as np
import pandas as pd
import time
import torch
import multiprocessing
import mrcfile
import urllib.request

# Configuration for Fourier masking reconstruction
default_result_dir = 'results/fourier_masking'
default_gpu_ids = [0, 1, 2, 3]
default_batch_size = 16
default_block_size = 1
default_num_masks = 1
default_mask_type = 'random_fourier'
default_mask_probs = [0.2, 0.3, 0.4, 0.5]
default_zeta_scale = 0.1
default_beta = 0.9

# Experiments collection
experiments = [
    {
        'protein': 'EMPIAR10076_128',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_128x128/resolve/main/EMPIAR10076_128x128_valset.pt'
    },
    # {
    #     'protein': 'EMPIAR11526_128',
    #     'model': 'anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128',
    #     'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR11526_128x128/resolve/main/EMPIAR11526_128x128_valset.mrc'
    # },
    # {
    #     'protein': 'EMPIAR10166_128',
    #     'model': 'anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128',
    #     'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10166_128x128/resolve/main/EMPIAR10166_128x128_valset.mrc'
    # },
    # {
    #     'protein': 'EMPIAR10786_128',
    #     'model': 'anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128',
    #    'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10786_128x128/resolve/main/EMPIAR10786_128x128_valset.mrc'
    # },

    # {
    #     'protein': 'EMPIAR10076_256',
    #     'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-256x256',
    #     'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_256x256/resolve/main/EMPIAR10076_256x256_valset.mrc'
    # },

    # {
    #     'protein': 'EMPIAR10648_256',
    #     'model': 'anonymousneurips008/empiar10648-ddpm-cryoem-256x256',
    #     'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10648_256x256/resolve/main/EMPIAR10648_256x256_valset.mrc'
    # },
    # Additional datasets can be added here
    # {
    #     'protein': 'dataset_name',
    #     'model': 'path/to/model',
    #     'val_dataset': 'path/to/dataset'
    # },
]

def download_dataset(url, save_path):
    """Download dataset from a URL to the specified save path"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading dataset from {url} to {save_path}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Download complete: {save_path}")
        return save_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def count_images_in_dataset(dataset_path):
    """Count the number of images in the dataset"""
    # If dataset is a URL, download it first
    if dataset_path.startswith('http://') or dataset_path.startswith('https://'):
        # Extract protein name from path (assuming path contains protein name)
        protein_name = os.path.basename(dataset_path).split('_')[0]
        data_dir = os.path.join('data', protein_name)
        os.makedirs(data_dir, exist_ok=True)

        # Get filename from URL
        filename = os.path.basename(dataset_path)
        local_path = os.path.join(data_dir, filename)

        # Download only if the file doesn't already exist
        if not os.path.exists(local_path):
            local_path = download_dataset(dataset_path, local_path)
            if local_path is None:
                print(f"Failed to download dataset, cannot count images")
                return 0
        else:
            print(f"Using existing downloaded dataset: {local_path}")

        # Update dataset path to use the local path
        dataset_path = local_path

    if dataset_path.endswith('.pt'):
        dataset = torch.load(dataset_path, map_location='cpu')
        return len(dataset)
    else:
        # For .mrcs files, use mrcfile to load the data
        with mrcfile.open(dataset_path) as mrc:
            dataset = mrc.data
        return len(dataset)

def run_gpu_job(gpu_id, start_id, end_id, experiment, mask_prob_result_dir,
               batch_size, block_size, num_masks, mask_type, mask_prob, zeta_scale, beta, use_config=True, verbose=False):
    """Run reconstruction for a range of images on a single GPU"""
    print(f"Starting reconstruction on GPU {gpu_id} for images {start_id}-{end_id} of {experiment['protein']} with mask_prob={mask_prob}")

    # Create log file for detailed output
    log_file = os.path.join(mask_prob_result_dir, f"gpu_{gpu_id}_log.txt")

    # Check if dataset is a URL, download it if necessary
    dataset_path = experiment['val_dataset']
    if dataset_path.startswith('http://') or dataset_path.startswith('https://'):
        protein_name = experiment['protein']
        # Create data directory for protein
        data_dir = os.path.join('data', protein_name)
        os.makedirs(data_dir, exist_ok=True)

        # Get filename from URL
        filename = os.path.basename(dataset_path)
        local_path = os.path.join(data_dir, filename)

        # Download only if the file doesn't already exist
        if not os.path.exists(local_path):
            local_path = download_dataset(dataset_path, local_path)
            if local_path is None:
                print(f"Failed to download dataset for {protein_name}, aborting job")
                return False
        else:
            print(f"Using existing downloaded dataset: {local_path}")

        # Update dataset path to use the local path
        dataset_path = local_path

    # Construct and run the command with output redirection
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'cryogen',
        '--model', experiment['model'],
        '--cryoem_path', dataset_path,
        '--start_id', str(start_id),
        '--end_id', str(end_id),
        '--result_dir', mask_prob_result_dir,
        '--batch_size', str(batch_size),
        '--block_size', str(block_size),
        '--num_masks', str(num_masks),
        '--mask_type', str(mask_type),
        '--mask_prob', str(mask_prob),
        '--zeta_scale', str(zeta_scale),
        '--beta', str(beta),
    ]

    if use_config:
        cmd.append('--use_config')

    if verbose:
        cmd.append('--verbose')

    cmd.append(f'> {log_file} 2>&1')

    print(f"GPU {gpu_id}: Started reconstruction for {experiment['protein']} images {start_id}-{end_id} with mask_prob={mask_prob}")

    # Run the command and periodically show brief progress
    process = subprocess.Popen(' '.join(cmd), shell=True)

    # Wait for the process while occasionally checking for progress
    while process.poll() is None:
        time.sleep(30)  # Check every 30 seconds
        print(f"GPU {gpu_id}: Still reconstructing {experiment['protein']} images {start_id}-{end_id} (mask_prob={mask_prob}) ({time.strftime('%H:%M:%S')})")

    if process.returncode == 0:
        print(f"GPU {gpu_id}: Completed reconstruction for {experiment['protein']} images {start_id}-{end_id} (mask_prob={mask_prob}) successfully")
    else:
        print(f"GPU {gpu_id}: Error during reconstruction for {experiment['protein']} images {start_id}-{end_id} (mask_prob={mask_prob}), return code: {process.returncode}")

    return True

def analyze_results(result_dir, block_size, num_masks, mask_type, mask_prob):
    """Analyze reconstruction results"""
    metrics_file = os.path.join(result_dir, 'reconstruction_metrics.csv')

    if not os.path.exists(metrics_file):
        print(f"Warning: No metrics file found in {result_dir}")
        return

    # Read metrics
    try:
        df = pd.read_csv(metrics_file)

        # Calculate average metrics
        avg_metrics = {
            'LPIPS': df['LPIPS'].mean() if 'LPIPS' in df.columns else float('nan'),
            'PSNR': df['PSNR'].mean() if 'PSNR' in df.columns else float('nan'),
            'SSIM': df['SSIM'].mean() if 'SSIM' in df.columns else float('nan'),
            'MSE': df['MSE'].mean() if 'MSE' in df.columns else float('nan'),
            'MAE': df['MAE'].mean() if 'MAE' in df.columns else float('nan'),
            'block_size': block_size,
            'num_masks': num_masks,
            'mask_type': mask_type,
            'mask_prob': mask_prob
        }

        print("\nReconstruction Results:")
        for metric, value in avg_metrics.items():
            if isinstance(value, (int, float)):
                print(f"Average {metric}: {value:.6f}")
            else:
                print(f"{metric}: {value}")

        # Save summary to file
        with open(os.path.join(result_dir, 'reconstruction_summary.json'), 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        return avg_metrics

    except Exception as e:
        print(f"Error reading metrics from {metrics_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run CryoGEN Fourier masking reconstruction with varying mask probabilities')
    parser.add_argument('--result_dir', type=str, default=default_result_dir, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size for processing')
    parser.add_argument('--block_size', type=int, default=default_block_size, help='Block size for downsampling (default: 1 for Fourier masking)')
    parser.add_argument('--num_masks', type=int, default=default_num_masks, help='Number of masks to use (default: 1 for Fourier masking)')
    parser.add_argument('--mask_type', type=str, default=default_mask_type, help='Type of mask to use (default: random_fourier)')
    parser.add_argument('--mask_probs', type=str, default=','.join(map(str, default_mask_probs)), help='Comma-separated list of mask probabilities (default: 0.1,0.2,0.3,0.4,0.5)')
    parser.add_argument('--zeta_scale', type=float, default=default_zeta_scale, help='Zeta scale parameter for gradient step size (default: 0.1)')
    parser.add_argument('--beta', type=float, default=default_beta, help='Beta parameter for momentum (default: 0.9)')
    parser.add_argument('--gpu_ids', type=str, default=','.join(map(str, default_gpu_ids)), help='Comma-separated list of GPU IDs')
    parser.add_argument('--use_config', action='store_true', help='Use configuration parameters based on block size')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output and detailed visualizations')
    parser.add_argument('--single_dataset', action='store_true', help='Run on a single dataset specified by command line args')
    parser.add_argument('--protein_name', type=str, help='Protein name (required if single_dataset=True)')
    parser.add_argument('--model', type=str, help='Model path (required if single_dataset=True)')
    parser.add_argument('--dataset', type=str, help='Dataset path (required if single_dataset=True)')
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
    print(f"Using GPUs: {gpu_ids}")

    # Parse mask probabilities
    mask_probs = [float(prob.strip()) for prob in args.mask_probs.split(',')]
    print(f"Testing mask probabilities: {mask_probs}")

    # Create main results directory
    os.makedirs(args.result_dir, exist_ok=True)

    # Define experiments to run
    if args.single_dataset:
        if not args.protein_name or not args.model or not args.dataset:
            parser.error("When using --single_dataset, you must specify --protein_name, --model, and --dataset")
        experiments_to_run = [{
            'protein': args.protein_name,
            'model': args.model,
            'val_dataset': args.dataset
        }]
    else:
        experiments_to_run = experiments

    print(f"Running with parameters: block_size={args.block_size}, num_masks={args.num_masks}, mask_type={args.mask_type}, zeta_scale={args.zeta_scale}, beta={args.beta}")

    # Process each experiment
    for experiment in experiments_to_run:
        protein_name = experiment['protein']
        protein_result_dir = os.path.join(args.result_dir, protein_name)
        os.makedirs(protein_result_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing {protein_name}")
        print(f"Model: {experiment['model']}")
        print(f"Dataset: {experiment['val_dataset']}")
        print(f"{'='*60}")

        # Count total images in dataset
        try:
            total_images = count_images_in_dataset(experiment['val_dataset'])
            print(f"Total images in dataset: {total_images}")
        except Exception as e:
            print(f"Error counting images in dataset: {e}")
            print("Using default value of 100 images")
            total_images = 100

        # Process each mask probability
        all_results = []
        for mask_prob in mask_probs:
            print(f"\n{'-'*60}")
            print(f"Running experiment with mask_prob = {mask_prob}")
            print(f"{'-'*60}")

            # Create result directory for this mask probability
            mask_prob_result_dir = os.path.join(protein_result_dir, f"maskprob_{mask_prob}")
            os.makedirs(mask_prob_result_dir, exist_ok=True)

            # Distribute images evenly across GPUs
            images_per_gpu = math.ceil(total_images / len(gpu_ids))

            # Launch one process per GPU
            processes = []
            for i, gpu_id in enumerate(gpu_ids):
                start_id = i * images_per_gpu
                if start_id >= total_images:
                    break

                end_id = min((i + 1) * images_per_gpu - 1, total_images - 1)

                # Create a separate process for each GPU
                process = multiprocessing.Process(
                    target=run_gpu_job,
                    args=(gpu_id, start_id, end_id, experiment, mask_prob_result_dir,
                         args.batch_size, args.block_size, args.num_masks, args.mask_type,
                         mask_prob, args.zeta_scale, args.beta, args.use_config, args.verbose)
                )
                processes.append(process)
                process.start()

                # Short delay to prevent GPU conflicts
                time.sleep(2)

            # Wait for all processes to complete
            for process in processes:
                process.join()

            print(f"Completed mask_prob = {mask_prob}")

            # Analyze results for this mask probability
            results = analyze_results(mask_prob_result_dir, args.block_size, args.num_masks, args.mask_type, mask_prob)
            if results:
                all_results.append(results)

        # Save comprehensive summary for all mask probabilities
        if all_results:
            summary_file = os.path.join(protein_result_dir, 'all_maskprob_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=4)

            print(f"\n{'='*60}")
            print(f"All Fourier masking experiments complete for {protein_name}!")
            print(f"{'='*60}")
            print("\nResults summary:")
            for mask_prob in mask_probs:
                mask_prob_result_dir = os.path.join(protein_result_dir, f"maskprob_{mask_prob}")
                print(f"  - mask_prob {mask_prob}: {mask_prob_result_dir}")
            print(f"\nComprehensive summary saved to: {summary_file}")
            print("Expected trend: Higher mask_prob -> Better reconstruction quality")

if __name__ == "__main__":
    main()
