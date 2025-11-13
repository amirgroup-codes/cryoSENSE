#!/usr/bin/env python3
"""
Run baseline reconstruction experiments for CryoGEN comparative methods.

This script performs:
- Random image selection for each protein to set as the validation set (same as CryoGEN)
- Grid search over reconstruction hyperparameters (for all methods not DMPlug) 
- Final reconstruction runs for each configuration 

DMPlug's configuration is set as 'superres', which performs super resolution experiments as from natural images.
"""
import os
import subprocess
import itertools
import torch
import random
import pandas as pd
import numpy as np
import json
import urllib.request

import pathlib as Path
cache_dir ="/usr/scratch/dtsui/.cache"
# Set up cache directories using user-provided cache directory
fake_home = os.path.join(cache_dir, "fake_home")
os.environ["HOME"] = fake_home
Path.home = lambda: Path(fake_home)
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------
# Experiment Configuration
# -------------------------------
block_sizes = [1]
undersampling_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mask_types = ['random_fourier','fourier_radial', 'fourier_ring']
noise_levels = [0.0, 0.1]  # No noise or moderate noise
baselines = ["wavelet", "tv_minimize", "dct"]  # Options: 'wavelet', 'tv_minimize', 'dct', 'dmplug'

# Grid search hyperparameters
lambdas = [1e-1, 1e-2, 1e-3, 1e-4]
learning_rates = [1e-3, 1e-2, 1e-1, 0.5, 1]

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

experiments = [
    # {
    #     'protein': 'EMPIAR10076_128',
    #     'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
    #     'train_dataset': f'{DATA_DIR}/EMPIAR10076_128x128_trainset.pt',
    #     'val_dataset': f'{DATA_DIR}/EMPIAR10076_128x128_valset.pt',
    #     'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_128x128/resolve/main/EMPIAR10076_128x128_trainset.pt',
    #     'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_128x128/resolve/main/EMPIAR10076_128x128_valset.pt'
    # },
    # {
    #     'protein': 'EMPIAR11526_128',
    #     'model': 'anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128',
    #     'train_dataset': f'{DATA_DIR}/EMPIAR11526_128x128_trainset.mrcs',
    #     'val_dataset': f'{DATA_DIR}/EMPIAR11526_128x128_valset.mrc',
    #     'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR11526_128x128/resolve/main/EMPIAR11526_128x128_trainset.mrcs',
    #     'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR11526_128x128/resolve/main/EMPIAR11526_128x128_valset.mrc'
    # },
    # {
    #     'protein': 'EMPIAR10166_128',
    #     'model': 'anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128',
    #     'train_dataset': f'{DATA_DIR}/EMPIAR10166_128x128_trainset.mrcs',
    #     'val_dataset': f'{DATA_DIR}/EMPIAR10166_128x128_valset.mrc',
    #     'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10166_128x128/resolve/main/EMPIAR10166_128x128_trainset.mrcs',
    #     'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10166_128x128/resolve/main/EMPIAR10166_128x128_valset.mrc'
    # },
    {
        'protein': 'EMPIAR10786_128',
        'model': 'anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128',
        'train_dataset': f'{DATA_DIR}/EMPIAR10786_128x128_trainset.mrcs',
        'val_dataset': f'{DATA_DIR}/EMPIAR10786_128x128_valset.mrc',
        'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10786_128x128/resolve/main/EMPIAR10786_128x128_trainset.mrcs',
        'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10786_128x128/resolve/main/EMPIAR10786_128x128_valset.mrc'
    },
    # {
    #     'protein': 'EMPIAR10076_256',
    #     'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-256x256',
    #     'train_dataset': f'{DATA_DIR}/EMPIAR10076_256x256_trainset.mrcs',
    #     'val_dataset': f'{DATA_DIR}/EMPIAR10076_256x256_valset.mrc',
    #     'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_256x256/resolve/main/EMPIAR10076_256x256_trainset.mrcs',
    #     'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10076_256x256/resolve/main/EMPIAR10076_256x256_valset.mrc'
    # },
    # {
    #     'protein': 'EMPIAR10648_256',
    #     'model': 'anonymousneurips008/empiar10648-ddpm-cryoem-256x256',
    #     'train_dataset': f'{DATA_DIR}/EMPIAR10648_256x256_trainset.mrcs',
    #     'val_dataset': f'{DATA_DIR}/EMPIAR10648_256x256_valset.mrc',
    #     'train_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10648_256x256/resolve/main/EMPIAR10648_256x256_trainset.mrcs',
    #     'val_url': 'https://huggingface.co/datasets/anonymousneurips008/EMPIAR10648_256x256/resolve/main/EMPIAR10648_256x256_valset.mrc'
    # },
]

# -------------------------------
# Utility Functions
# -------------------------------

def download_dataset(url, save_path):
    """Download dataset from a URL to the specified save path"""
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading dataset from {url} to {save_path}")
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Download complete: {save_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    else:
        print(f"Dataset already exists at {save_path}")

def select_random_images(dataset_path, num_images=16, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dataset_path.endswith('.mrcs') or dataset_path.endswith('.mrc'):
        import mrcfile
        dataset = torch.from_numpy(mrcfile.open(dataset_path).data.copy())
    else:
        dataset = torch.load(dataset_path)
    indices = np.random.choice(len(dataset), size=num_images, replace=False)
    return torch.stack([dataset[i] for i in indices])

def save_selected_images(dataset_path, result_path, num_images=16):
    selected = select_random_images(dataset_path, num_images)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    torch.save(selected, result_path)
    return result_path

def run_subprocess(cmd, log_file):
    process = subprocess.Popen(' '.join(cmd), shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with code {process.returncode}")
    return process.returncode

# -------------------------------
# Main Experiment Logic
# -------------------------------

def run_experiments():
    for exp in experiments:
        protein = exp['protein']
        train_path = exp['train_dataset']
        val_path = exp['val_dataset']
        model_path = exp['model']

        # Download datasets if needed
        if 'train_url' in exp:
            download_dataset(exp['train_url'], train_path)
        if 'val_url' in exp:
            download_dataset(exp['val_url'], val_path)

        print('----------------------')
        print(f"Running experiments for {protein}")
        print('----------------------')

        os.makedirs(f'results/{protein}', exist_ok=True)
        selected_val_path = save_selected_images(val_path, f'results/{protein}/selected_images.pt')

        all_params = list(itertools.product(block_sizes, undersampling_factors, mask_types))

        for block_size, mask_prob, mask_type in all_params:
            for baseline in baselines:
                print(f"  {baseline} - block: {block_size}, prob: {mask_prob}, type: {mask_type}")
                for noise_level in noise_levels:

                    if baseline == 'dmplug':
                        mask_type = 'random_fourier'

                    result_dir = os.path.join(f'results_{noise_level}' if noise_level > 0 else 'results',
                                            protein, baseline,
                                            f"block_{block_size}_prob_{mask_prob}_{mask_type}")
                    os.makedirs(result_dir, exist_ok=True)

                    if baseline != 'dmplug':
                        grid_dir = os.path.join('grid_search', protein, baseline,
                                                f"block_{block_size}_prob_{mask_prob}_{mask_type}")
                        os.makedirs(grid_dir, exist_ok=True)
                        result_dir_no_noise = os.path.join('results',
                                            protein, baseline,
                                            f"block_{block_size}_prob_{mask_prob}_{mask_type}")
                        best_params_path = os.path.join(result_dir_no_noise, 'best_params.json')
                        if not os.path.exists(best_params_path):
                            print('Performing grid search for lambda and learning rate')
                            selected_train_path = save_selected_images(train_path, os.path.join(grid_dir, 'selected_images.pt'), num_images=2)
                            grid_results = []

                            for lmda, lr in itertools.product(lambdas, learning_rates):
                                print(f'lambda_{lmda}_lr_{lr}')
                                sub_dir = os.path.join(grid_dir, f"lambda_{lmda}_lr_{lr}")
                                os.makedirs(sub_dir, exist_ok=True)
                                log = os.path.join(sub_dir, 'experiment_log.txt')
                                cmd = [
                                    'python', 'baselines.py',
                                    '--use_cryoem',
                                    '--cryoem_path', selected_train_path,
                                    '--result_dir', sub_dir,
                                    '--block_size', str(block_size),
                                    '--num_masks', '1',
                                    '--mask_prob', str(mask_prob),
                                    '--mask_type', mask_type,
                                    '--baseline', baseline,
                                    '--lambda_', str(lmda),
                                    '--learning_rate', str(lr),
                                    '--batch_size', '16',
                                    '--end_id', '1',
                                    '--model', model_path,
                                    f'> {log} 2>&1'
                                ]
                                if run_subprocess(cmd, log) == 0:
                                    df = pd.read_csv(os.path.join(sub_dir, "reconstruction_metrics.csv"))
                                    grid_results.append({
                                        'lambda': lmda,
                                        'lr': lr,
                                        'LPIPS': df['LPIPS'].mean(),
                                        'SSIM': df['SSIM'].mean(),
                                        'PSNR': df['PSNR'].mean()
                                    })

                            best_row = sorted(grid_results, key=lambda x: x['LPIPS'])[0]
                            with open(best_params_path, 'w') as f:
                                json.dump({'lambda': best_row['lambda'], 'lr': best_row['lr']}, f)

                    # Load best params
                    if baseline == 'dmplug':
                        lmda = 0
                        lr = 0
                        mask_type = 'random_fourier'
                    else:
                        print('Loading best params from best_params.json')
                        with open(os.path.join(result_dir_no_noise, 'best_params.json')) as f:
                            hp = json.load(f)
                        lmda = hp['lambda']
                        lr = hp['lr']

                    # result_dir = os.path.join(f'results_{noise_level}' if noise_level > 0 else 'results',
                    #     protein, baseline,
                    #     f"block_{block_size}_prob_{mask_prob}_{mask_type}_run{SEED}")
                    # os.makedirs(result_dir, exist_ok=True)
                    out_path = os.path.join(result_dir, "reconstruction_raw_image_0.pt")

                    if os.path.exists(out_path):
                        print('Noise level:', noise_level)
                        log = os.path.join(result_dir, 'experiment_log.txt')
                        cmd = [
                            'python', 'baselines.py',
                            '--use_cryoem',
                            '--cryoem_path', selected_val_path,
                            '--result_dir', result_dir,
                            '--block_size', str(block_size),
                            '--num_masks', '1',
                            '--mask_prob', str(mask_prob),
                            '--mask_type', mask_type,
                            '--baseline', baseline,
                            '--lambda_', str(lmda),
                            '--learning_rate', str(lr),
                            '--batch_size', '16',
                            '--end_id', '15',
                            '--noise_level', str(noise_level),
                            '--model', model_path,
                            f'> {log} 2>&1'
                        ]
                        run_subprocess(cmd, log)

if __name__ == "__main__":
    run_experiments()
