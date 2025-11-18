#!/usr/bin/env python3
"""
simple_example.py

A simple example script to demonstrate pixel-space and Fourier-space masking
"""

import os
import subprocess
import torch
import numpy as np
import urllib.request
import pathlib as Path
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

cache_dir = "/usr/scratch/dtsui/.cache"
fake_home = os.path.join(cache_dir, "fake_home")
os.environ["HOME"] = fake_home
Path.home = lambda: Path(fake_home)

# -------------------------------
# Pixel-space masking
# -------------------------------
CONFIG = {
    'protein': 'EMPIAR10076_128',
    'prior': 'dct',
    'block_size': 2, # <---- determines compression factor
    'num_masks': 2,  # <---- determines compression factor
    'mask_type': 'random_binary',       
}
RESULTS_DIR = 'simple_example_pixel'
run_name = 'pixel_masking'
os.makedirs(RESULTS_DIR, exist_ok=True)
out_dir = os.path.join(RESULTS_DIR, run_name)
os.makedirs(out_dir, exist_ok=True)
log_file = os.path.join(out_dir, 'run.log')

cmd = [
    'python', 'main.py',
    '--use_cryoem',
    '--cryoem_path', '../data/sample_empiar10076.pt',
    '--result_dir', out_dir,
    '--block_size', str(CONFIG['block_size']), 
    '--num_masks', str(CONFIG['num_masks']),   
    '--mask_type', CONFIG['mask_type'],
    '--prior', CONFIG['prior'],
    '--lambda_', '1e-2',
    '--learning_rate', '1e-1',
    '--start_id', '0', 
    '--end_id', '0', 
    '--noise_level', '0',
]

with open(log_file, 'w') as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    process.wait()
if process.returncode == 0:
    print(f'\nPixel-space masking results saved to {out_dir}')
    print(f'Compression factor: {CONFIG["block_size"]**2 / CONFIG["num_masks"]}')
else:
    print(f"\nError! Process failed with code {process.returncode}. Check {log_file}")



# -------------------------------
# Fourier-space masking
# -------------------------------
CONFIG = {
    'protein': 'EMPIAR10076_128',
    'prior': 'dct',
    'block_size': 1,
    'num_masks': 1,
    'mask_prob': 0.5, # <---- determines compression factor, num_masks and block_size are left as 1
    'mask_type': 'random_fourier',       
}
RESULTS_DIR = 'simple_example_fourier'
run_name = 'fourier_masking'
os.makedirs(RESULTS_DIR, exist_ok=True)
out_dir = os.path.join(RESULTS_DIR, run_name)
os.makedirs(out_dir, exist_ok=True)
log_file = os.path.join(out_dir, 'run.log')

cmd = [
    'python', 'main.py',
    '--use_cryoem',
    '--cryoem_path', '../data/sample_empiar10076.pt',
    '--result_dir', out_dir,
    '--block_size', str(CONFIG['block_size']), 
    '--num_masks', str(CONFIG['num_masks']),   
    '--mask_type', CONFIG['mask_type'],
    '--mask_prob', str(CONFIG['mask_prob']),
    '--prior', CONFIG['prior'],
    '--lambda_', '1e-2',
    '--learning_rate', '1e-1',
    '--start_id', '0', 
    '--end_id', '0', 
    '--noise_level', '0',
]

with open(log_file, 'w') as f:
    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    process.wait()
if process.returncode == 0:
    print(f'\nFourier-space masking results saved to {out_dir}')
    print(f'Compression factor: {1/CONFIG["mask_prob"]}')
else:
    print(f"\nError! Process failed with code {process.returncode}. Check {log_file}")