#!/usr/bin/env python3
"""
Script for running experiments with different parameters for CryoGEN.
Performs experiments across block sizes, number of masks, mask types, and noise levels and measures the performance 
"""

import itertools
import os
import torch
import numpy as np
import random
import subprocess
import threading
import queue
import pandas as pd
import json
import time
import argparse
import mrcfile
import urllib.request

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Default parameters for experiments
default_block_sizes = [2, 4, 8, 16, 32]
default_max_masks_per_block = [4, 16, 64, 256, 1024]  # Maximum masks for each block size
default_undersampling_factors = [0.25, 0.1, 0.1, 0.1, 0.1]  # Factor for each block size
default_mask_types = ['random_binary']
default_noise_levels = [0.0, 0.1]  # No noise or moderate noise
default_gpu_ids = [0]

# Default datasets
datasets = [
    {
        'protein': 'EMPIAR10076_128',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_128x128/resolve/main/EMPIAR10076_128x128_valset.pt'
    },
    {
        'protein': 'EMPIAR11526_128',
        'model': 'anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR11526_128x128/resolve/main/EMPIAR11526_128x128_valset.mrc'
    },
    {
        'protein': 'EMPIAR10166_128',
        'model': 'anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10166_128x128/resolve/main/EMPIAR10166_128x128_valset.mrc'
    },
    {
        'protein': 'EMPIAR10786_128',
        'model': 'anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128',
       'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10786_128x128/resolve/main/EMPIAR10786_128x128_valset.mrc'
    },

    {
        'protein': 'EMPIAR10076_256',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-256x256',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_256x256/resolve/main/EMPIAR10076_256x256_valset.mrc'
    },

    {
        'protein': 'EMPIAR10648_256',
        'model': 'anonymousneurips008/empiar10648-ddpm-cryoem-256x256',
        'val_dataset': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10648_256x256/resolve/main/EMPIAR10648_256x256_valset.mrc'
    },
]

def select_random_images(dataset_path, num_images, seed=42):
    """
    Load dataset and select random images with fixed seed
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset based on file extension
    if dataset_path.endswith('.mrcs'):
        try:
            with mrcfile.open(dataset_path) as mrc:
                dataset = torch.from_numpy(mrc.data)
        except Exception as e:
            print(f"Error loading .mrcs file: {e}")
            return None
    else:  # Assume .pt file
        try:
            dataset = torch.load(dataset_path, map_location='cpu')
        except Exception as e:
            print(f"Error loading .pt file: {e}")
            return None
    
    # Ensure dataset has enough images
    if len(dataset) < num_images:
        print(f"Warning: Dataset has only {len(dataset)} images, but {num_images} were requested.")
        num_images = len(dataset)
    
    # Randomly select images
    indices = np.random.choice(len(dataset), size=num_images, replace=False)
    selected = torch.stack([dataset[i] for i in indices])

    return selected

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

def prepare_dataset(experiment, result_dir, num_images=16):
    """Prepare dataset by selecting random images and saving them for experiment"""
    protein_name = experiment['protein']
    val_dataset = experiment['val_dataset']
    
    # Create result directory if it doesn't exist
    os.makedirs(os.path.join(result_dir, protein_name), exist_ok=True)
    
    # If dataset is a URL, download it first
    if val_dataset.startswith('http://') or val_dataset.startswith('https://'):
        # Create data directory for protein
        data_dir = os.path.join('data', protein_name)
        os.makedirs(data_dir, exist_ok=True)
        
        # Get filename from URL
        filename = os.path.basename(val_dataset)
        local_path = os.path.join(data_dir, filename)
        
        # Download only if the file doesn't already exist
        if not os.path.exists(local_path):
            local_path = download_dataset(val_dataset, local_path)
            if local_path is None:
                print(f"Failed to download dataset for {protein_name}")
                return None
        else:
            print(f"Using existing downloaded dataset: {local_path}")
        
        # Update val_dataset to use the local path
        val_dataset = local_path
    
    # Select random images from the dataset
    selected_images = select_random_images(val_dataset, num_images)
    if selected_images is None:
        print(f"Error preparing dataset for {protein_name}")
        return None
    
    # Save selected images for experiment
    save_path = os.path.join(result_dir, protein_name, 'selected_images.pt')
    torch.save(selected_images, save_path)
    print(f"Saved {num_images} selected images for {protein_name} to {save_path}")
    
    return save_path

def run_experiment(job):
    """Run a single experiment"""
    experiment, mask_type, block_size, num_masks, noise_level, gpu_id, result_dir = job
    
    protein_name = experiment['protein']
    model_name = experiment['model']
    selected_images_path = experiment['selected_images']
    
    # Create a specific result directory for this configuration
    config_dir = os.path.join(result_dir, protein_name, f"block_{block_size}_masks_{num_masks}_{mask_type}_noise_{noise_level}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Log file for detailed output
    log_file = os.path.join(config_dir, "experiment_log.txt")
    
    print(f"Running experiment: {protein_name}, block_size={block_size}, num_masks={num_masks}, mask_type={mask_type}, noise_level={noise_level} on GPU {gpu_id}")
    
    # Construct command to run cryogen with appropriate parameters
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'cryogen',
        '--model', model_name,
        '--cryoem_path', selected_images_path,
        '--block_size', str(block_size),
        '--num_masks', str(num_masks),
        '--mask_type', mask_type,
        '--noise_level', str(noise_level),
        '--result_dir', config_dir,
        '--batch_size', '4',  # Small batch size for experiment
        '--start_id', '0',
        '--end_id', '15',  # Assuming 16 images
        '--use_config',  # Use configuration file based on block size
        f'> {log_file} 2>&1'
    ]
    
    # Run the command
    process = subprocess.Popen(' '.join(cmd), shell=True)
    
    # Wait for the process while periodically checking progress
    while process.poll() is None:
        time.sleep(30)  # Check every 30 seconds
        print(f"GPU {gpu_id}: {protein_name}, block={block_size}, masks={num_masks}, still running ({time.strftime('%H:%M:%S')})")
    
    if process.returncode == 0:
        print(f"Experiment completed successfully: {protein_name}, block={block_size}, masks={num_masks}")
    else:
        print(f"Experiment failed with return code {process.returncode}: {protein_name}, block={block_size}, masks={num_masks}")

def gpu_worker(gpu_id, job_queue, result_dir):
    """Worker function that processes jobs from the queue on a specific GPU"""
    while True:
        try:
            job = job_queue.get(block=False)
            if job is None:  # Sentinel value to exit
                break
                
            # Add the GPU ID and result_dir to the job
            job_with_info = job + (gpu_id, result_dir)
            run_experiment(job_with_info)
            
            # Mark job as done
            job_queue.task_done()
        except queue.Empty:
            # No more jobs in the queue
            break

def analyze_results(result_dir):
    """Analyze results from all experiments"""
    all_results = []
    
    # Iterate through all protein directories
    for protein_dir in os.listdir(result_dir):
        protein_path = os.path.join(result_dir, protein_dir)
        if not os.path.isdir(protein_path) or protein_dir == 'best_configs':
            continue
            
        # Iterate through configuration directories
        for config_dir in os.listdir(protein_path):
            if config_dir == 'selected_images.pt':
                continue
                
            if not config_dir.startswith('block_'):
                continue
                
            config_path = os.path.join(protein_path, config_dir)
            if not os.path.isdir(config_path):
                continue
                
            # Parse configuration from directory name
            try:
                config_parts = config_dir.split('_')
                block_size = int(config_parts[1])
                num_masks = int(config_parts[3])
                mask_type = config_parts[4]
                noise_level = float(config_parts[6]) if len(config_parts) > 6 else 0.0
            except (ValueError, IndexError) as e:
                print(f"Error parsing config from {config_dir}: {e}")
                continue
            
            # Check for metrics file
            metrics_file = os.path.join(config_path, 'reconstruction_metrics.csv')
            if not os.path.exists(metrics_file):
                print(f"No metrics file in {config_path}")
                continue
            
            # Read metrics
            try:
                df = pd.read_csv(metrics_file)
                avg_psnr = df['PSNR'].mean() if 'PSNR' in df.columns else float('nan')
                avg_ssim = df['SSIM'].mean() if 'SSIM' in df.columns else float('nan')
                avg_lpips = df['LPIPS'].mean() if 'LPIPS' in df.columns else float('nan')
                avg_mse = df['MSE'].mean() if 'MSE' in df.columns else float('nan')
                
                # Store results
                result = {
                    'protein': protein_dir,
                    'block_size': block_size,
                    'num_masks': num_masks,
                    'mask_type': mask_type,
                    'noise_level': noise_level,
                    'avg_psnr': avg_psnr,
                    'avg_ssim': avg_ssim,
                    'avg_lpips': avg_lpips,
                    'avg_mse': avg_mse,
                }
                
                all_results.append(result)
                
                # Save individual summary
                with open(os.path.join(config_path, 'summary.json'), 'w') as f:
                    json.dump(result, f, indent=4)
                
            except Exception as e:
                print(f"Error analyzing metrics in {metrics_file}: {e}")
    
    # Save all results
    if all_results:
        # Save as JSON
        with open(os.path.join(result_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
        
        # Save as CSV for easier analysis
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(result_dir, 'all_results.csv'), index=False)
        
        # Find best configurations
        best_configs = {}
        for protein in df['protein'].unique():
            protein_df = df[df['protein'] == protein]
            
            # Get best configs based on different metrics
            try:
                best_psnr_idx = protein_df['avg_psnr'].idxmax()
                best_ssim_idx = protein_df['avg_ssim'].idxmax()
                best_lpips_idx = protein_df['avg_lpips'].idxmin()  # Lower is better
                
                best_configs[protein] = {
                    'best_psnr': protein_df.loc[best_psnr_idx].to_dict(),
                    'best_ssim': protein_df.loc[best_ssim_idx].to_dict(),
                    'best_lpips': protein_df.loc[best_lpips_idx].to_dict()
                }
            except Exception as e:
                print(f"Error finding best config for {protein}: {e}")
        
        # Create directory for best configs
        best_dir = os.path.join(result_dir, 'best_configs')
        os.makedirs(best_dir, exist_ok=True)
        
        # Save best configs
        with open(os.path.join(best_dir, 'best_configs.json'), 'w') as f:
            json.dump(best_configs, f, indent=4)
        
        print("\nBest Configurations:")
        for protein, configs in best_configs.items():
            print(f"\n{protein}:")
            print(f"  Best PSNR: block_size={configs['best_psnr']['block_size']}, num_masks={configs['best_psnr']['num_masks']}, PSNR={configs['best_psnr']['avg_psnr']:.2f}")
            print(f"  Best SSIM: block_size={configs['best_ssim']['block_size']}, num_masks={configs['best_ssim']['num_masks']}, SSIM={configs['best_ssim']['avg_ssim']:.4f}")
            print(f"  Best LPIPS: block_size={configs['best_lpips']['block_size']}, num_masks={configs['best_lpips']['num_masks']}, LPIPS={configs['best_lpips']['avg_lpips']:.4f}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Run experiments for CryoGEN')
    parser.add_argument('--result_dir', type=str, default='experiment_results_2d_plots',
                       help='Directory to save experiment results')
    parser.add_argument('--gpu_ids', type=str, default=','.join(map(str, default_gpu_ids)),
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--single_protein', action='store_true',
                       help='Run experiment on a single protein')
    parser.add_argument('--protein_name', type=str,
                       help='Name of the protein (used with --single_protein)')
    parser.add_argument('--model', type=str,
                       help='Path to the model (used with --single_protein)')
    parser.add_argument('--dataset', type=str,
                       help='Path to the dataset (used with --single_protein)')
    parser.add_argument('--num_images', type=int, default=16,
                       help='Number of images to select for experiment')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze existing results without running experiments')
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    
    # Create result directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Define experiments
    if args.single_protein:
        if not args.protein_name or not args.model or not args.dataset:
            parser.error("With --single_protein, you must specify --protein_name, --model, and --dataset")
        
        # Single protein experiment
        experiments = [{
            'protein': args.protein_name,
            'model': args.model,
            'val_dataset': args.dataset
        }]
    else:
        # Use the default datasets
        experiments = datasets
    
    # If only analyzing results, skip to analysis
    if args.analyze_only:
        print(f"Analyzing existing results in {args.result_dir}")
        analyze_results(args.result_dir)
        return
    
    # Prepare datasets (select random images)
    prepared_experiments = []
    for experiment in experiments:
        selected_images_path = prepare_dataset(experiment, args.result_dir, args.num_images)
        if selected_images_path:
            exp_copy = experiment.copy()
            exp_copy['selected_images'] = selected_images_path
            prepared_experiments.append(exp_copy)
    
    if not prepared_experiments:
        print("No valid experiments to run")
        return
    
    # Generate parameter combinations for experiments
    parameter_combinations = []
    for block_idx, block_size in enumerate(default_block_sizes):
        # Get corresponding factors for this block size
        if block_idx < len(default_max_masks_per_block) and block_idx < len(default_undersampling_factors):
            max_masks = default_max_masks_per_block[block_idx]
            undersample_factor = default_undersampling_factors[block_idx]
            
            # Calculate number of masks at different sampling rates
            # For each block size, we'll test with different numbers of masks
            mask_counts = [int(max_masks * undersample_factor * i) for i in range(1, int(1/undersample_factor) + 1)]
            mask_counts = [count for count in mask_counts if count > 0]  # Ensure no zero counts
            
            # Create combinations of parameters
            for num_masks, mask_type, noise_level in itertools.product(
                mask_counts, default_mask_types, default_noise_levels):
                parameter_combinations.append((block_size, num_masks, mask_type, noise_level))
    
    print(f"Total number of parameter combinations: {len(parameter_combinations)}")
    print(f"Total number of experiments: {len(prepared_experiments) * len(parameter_combinations)}")
    
    # Create job queue
    job_queue = queue.Queue()
    
    # Add all experiments to the queue
    for experiment in prepared_experiments:
        for block_size, num_masks, mask_type, noise_level in parameter_combinations:
            job_queue.put((experiment, mask_type, block_size, num_masks, noise_level))
    
    # Start worker threads for each GPU
    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=gpu_worker, args=(gpu_id, job_queue, args.result_dir))
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete
    for thread in threads:
        thread.join()
    
    print("All experiments completed")
    
    # Analyze results
    analyze_results(args.result_dir)
    

if __name__ == "__main__":
    main() 