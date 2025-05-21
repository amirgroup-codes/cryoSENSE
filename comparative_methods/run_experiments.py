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

# -------------------------------
# Experiment Configuration
# -------------------------------
block_sizes = [2]
max_masks_per_block = [1]
undersampling_factors_per_block = [1]
mask_types = ['random_binary']
noise_levels = [0]#, 0.1]
baselines = ['dct', 'wavelet', 'tv_minimize', 'dmplug']

# block_sizes = [2, 4, 8, 16, 32]
# max_masks_per_block = [4, 16, 64, 256, 1024]  # Maximum masks for each block size
# undersampling_factors_per_block = [0.25, 0.1, 0.1, 0.1, 0.1]  # Factor for each block size
# mask_types = ['random_binary']
# noise_levels = [0.0, 0.1]  # No noise or moderate noise
# baselines = ['dct']  # Options: 'wavelet', 'tv_minimize', 'dct', 'dmplug'

# Grid search hyperparameters
lambdas = [1e-1, 1e-2]#, 1e-3, 1e-4]
learning_rates = [0.5, 1] #[1e-3]#, 1e-2, 1e-1, 0.5, 1]

experiments = [

    # Make sure to download the data from the links included in the README.md file and update the paths belo.
    # {
    #     'protein': 'EMPIAR10076_128',
    #     'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10076/data/data_treated_128/train.mrcs'
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10076/data/data_treated_128/val.mrcs'
    # },
    # {
    #     'protein': 'EMPIAR11526_128',
    #     'model': 'anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar11526/data/data_treated_128/train.mrcs',
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar11526/data/data_treated_128/val.mrcs'
    # },
    # {
    #     'protein': 'EMPIAR10166_128',
    #     'model': 'anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10166/data/data_treated_128/train.mrcs',
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10166/particles_val_normalized.mrcs'
    # },
    # {
    #     'protein': 'EMPIAR10786_128',
    #     'model': 'anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10786/data/data_treated_128/train.mrcs',
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10786/particles_val_normalized.mrcs'
    # },
    # {
    #     'protein': 'EMPIAR10076_256',
    #     'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-256x256',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10076/data/data_treated_256/train.mrcs',
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10076/diffusion/data_256/empiar10076_raw_val_test_256_normalized.mrcs'
    # },
    # {
    #     'protein': 'EMPIAR10648_256',
    #     'model': 'anonymousneurips008/empiar10648-ddpm-cryoem-256x256',
    #     'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar106486/data/data_treated_256/train.mrcs',
    #     'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10648/particles_val_bicubic_256_normalized.mrcs'
    # },

    # Testing
    {
        'protein': 'EMPIAR10786_128',
        'model': '/usr/scratch/danial_stuff/FrugalCryo/Test/training/ddpm-ema-cryoem-128-EMPIAR10786-apr16/',
        'train_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10786/particles_train_normalized.mrcs',
        'val_dataset': '/usr/scratch/CryoEM/CryoSensing/empiar10786/particles_val_normalized.mrcs'
    }
]

# -------------------------------
# Utility Functions
# -------------------------------

def select_random_images(dataset_path, num_images=16, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dataset_path.endswith('.mrcs'):
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

        print('----------------------')
        print(f"Running experiments for {protein}")
        print('----------------------')

        os.makedirs(f'results/{protein}', exist_ok=True)
        selected_val_path = save_selected_images(val_path, f'results/{protein}/selected_images.pt')

        all_params = []
        for idx, block_size in enumerate(block_sizes):
            max_masks = max_masks_per_block[idx]
            factor = undersampling_factors_per_block[idx]
            mask_counts = sorted(set(max(1, int(i * factor * max_masks)) for i in range(1, int(1/factor)+1)))
            all_params += list(itertools.product([block_size], mask_counts, mask_types))

        for block_size, num_mask, mask_type in all_params:
            for baseline in baselines:
                print(f"  {baseline} - block: {block_size}, masks: {num_mask}, type: {mask_type}")
                for noise_level in noise_levels:

                    if baseline == 'dmplug':
                        mask_type = 'superres'

                    result_dir = os.path.join(f'results_{noise_level}' if noise_level > 0 else 'results',
                                            protein, baseline,
                                            f"block_{block_size}_masks_{num_mask}_{mask_type}")
                    os.makedirs(result_dir, exist_ok=True)

                    if baseline != 'dmplug':
                        grid_dir = os.path.join('grid_search', protein, baseline,
                                                f"block_{block_size}_masks_{num_mask}_{mask_type}")
                        os.makedirs(grid_dir, exist_ok=True)
                        result_dir_no_noise = os.path.join('results',
                                            protein, baseline,
                                            f"block_{block_size}_masks_{num_mask}_{mask_type}")
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
                                    '--num_masks', str(num_mask),
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
                        mask_type = 'superres'
                    else:
                        print('Loading best params from best_params.json')
                        with open(os.path.join(result_dir_no_noise, 'best_params.json')) as f:
                            hp = json.load(f)
                        lmda = hp['lambda']
                        lr = hp['lr']

                    out_path = os.path.join(result_dir, "reconstruction_raw_image_0.pt")

                    if not os.path.exists(out_path):
                        print('Noise level:', noise_level)
                        log = os.path.join(result_dir, 'experiment_log.txt')
                        cmd = [
                            'python', 'baselines.py',
                            '--use_cryoem',
                            '--cryoem_path', selected_val_path,
                            '--result_dir', result_dir,
                            '--block_size', str(block_size),
                            '--num_masks', str(num_mask),
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
