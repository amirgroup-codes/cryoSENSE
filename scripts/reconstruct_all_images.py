#!/usr/bin/env python3
"""
Script for reconstructing multiple CryoEM images across different datasets,
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

# Default settings
default_result_dir = 'results/all_images'
default_gpu_ids = [0, 1, 2, 3]
default_batch_size = 16
default_block_size = 32
default_num_masks = 1024
default_mask_type = 'random_binary'

# Dataset collection
experiments = [
    {
        'protein': 'empiar10076_128',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
        'val_dataset': '/usr/scratch/danial_stuff/FrugalCryo/Test/new_pipeline/combined/empiar10076_128_val_test_combined.pt'
    },
    # Additional datasets can be added here
    # {
    #     'protein': 'dataset_name',
    #     'model': 'path/to/model',
    #     'val_dataset': 'path/to/dataset'
    # },
]

def count_images_in_dataset(dataset_path):
    """Count the number of images in the dataset"""
    if dataset_path.endswith('.pt'):
        dataset = torch.load(dataset_path, map_location='cpu')
        return len(dataset)
    else:
        # For .mrcs files, use mrcfile to load the data
        with mrcfile.open(dataset_path) as mrc:
            dataset = mrc.data
        return len(dataset)

def run_gpu_job(gpu_id, start_id, end_id, experiment, protein_result_dir, 
               batch_size, block_size, num_masks, mask_type, use_config=True, verbose=False):
    """Run reconstruction for a range of images on a single GPU"""
    print(f"Starting reconstruction on GPU {gpu_id} for images {start_id}-{end_id} of {experiment['protein']}")
    
    # Create log file for detailed output
    log_file = os.path.join(protein_result_dir, f"gpu_{gpu_id}_log.txt")
    
    # Construct and run the command with output redirection
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'cryogen',
        '--model', experiment['model'],
        '--cryoem_path', experiment['val_dataset'],
        '--start_id', str(start_id),
        '--end_id', str(end_id),
        '--result_dir', protein_result_dir,
        '--batch_size', str(batch_size),
        '--block_size', str(block_size),
        '--num_masks', str(num_masks),
        '--mask_type', str(mask_type),
    ]
    
    if use_config:
        cmd.append('--use_config')
    
    if verbose:
        cmd.append('--verbose')
    
    cmd.append(f'> {log_file} 2>&1')
    
    print(f"GPU {gpu_id}: Started reconstruction for {experiment['protein']} images {start_id}-{end_id}")
    
    # Run the command and periodically show brief progress
    process = subprocess.Popen(' '.join(cmd), shell=True)
    
    # Wait for the process while occasionally checking for progress
    while process.poll() is None:
        time.sleep(30)  # Check every 30 seconds
        print(f"GPU {gpu_id}: Still reconstructing {experiment['protein']} images {start_id}-{end_id} ({time.strftime('%H:%M:%S')})")
    
    if process.returncode == 0:
        print(f"GPU {gpu_id}: Completed reconstruction for {experiment['protein']} images {start_id}-{end_id} successfully")
    else:
        print(f"GPU {gpu_id}: Error during reconstruction for {experiment['protein']} images {start_id}-{end_id}, return code: {process.returncode}")
    
    return True

def analyze_results(result_dir, block_size, num_masks, mask_type):
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
            'mask_type': mask_type
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
    parser = argparse.ArgumentParser(description='Run CryoGEN reconstruction on multiple images')
    parser.add_argument('--result_dir', type=str, default=default_result_dir, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=default_batch_size, help='Batch size for processing')
    parser.add_argument('--block_size', type=int, default=default_block_size, help='Block size for downsampling')
    parser.add_argument('--num_masks', type=int, default=default_num_masks, help='Number of masks to use')
    parser.add_argument('--mask_type', type=str, default=default_mask_type, help='Type of mask to use')
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
    
    print(f"Running with parameters: block_size={args.block_size}, num_masks={args.num_masks}, mask_type={args.mask_type}")
    
    # Process each experiment
    for experiment in experiments_to_run:
        protein_name = experiment['protein']
        protein_result_dir = os.path.join(args.result_dir, protein_name)
        os.makedirs(protein_result_dir, exist_ok=True)
        
        print(f"\nProcessing {protein_name}")
        print(f"Model: {experiment['model']}")
        print(f"Dataset: {experiment['val_dataset']}")
        
        # Count total images in dataset
        try:
            total_images = count_images_in_dataset(experiment['val_dataset'])
            print(f"Total images in dataset: {total_images}")
        except Exception as e:
            print(f"Error counting images in dataset: {e}")
            print("Using default value of 100 images")
            total_images = 100
        
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
                args=(gpu_id, start_id, end_id, experiment, protein_result_dir, 
                     args.batch_size, args.block_size, args.num_masks, args.mask_type, 
                     args.use_config, args.verbose)
            )
            processes.append(process)
            process.start()
            
            # Short delay to prevent GPU conflicts
            time.sleep(2)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        print(f"All reconstruction processes completed for {protein_name}")
        
        # Analyze results for this protein
        analyze_results(protein_result_dir, args.block_size, args.num_masks, args.mask_type)

if __name__ == "__main__":
    main() 