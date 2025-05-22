#!/usr/bin/env python3
"""
Reconstruct all validation images using comparative methods.

Performs chunked reconstruction in parallel across multiple GPUs.
Automatically resumes from incomplete runs and uses hyperparameters from previous grid searches.

Steps:
- Loads validation dataset
- Splits into chunks (batch size per process)
- For each baseline + config combo:
    - Loads best hyperparameters (if applicable)
    - Spawns parallel jobs to reconstruct all images in chunks
"""

import os
import subprocess
import json
import torch
import time
from multiprocessing import Process

# -------------------------------
# Configuration
# -------------------------------

baseline = 'dct'
chunk_size = 16
gpu_ids = [1]
num_parallel = len(gpu_ids)

# Configurations
configs = [
    {'block_size': 2, 'num_masks': 1, 'mask_type': 'random_binary'},
]

# Dataset settings (assumes downloaded datasets from HuggingFace)
data_dir = 'data'
experiments = [
    {
        'protein': 'EMPIAR10076_128',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128',
        'val_dataset': f'{data_dir}/EMPIAR10076_128x128_valset.pt'
    },
    {
        'protein': 'EMPIAR11526_128',
        'model': 'anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128',
        'val_dataset': f'{data_dir}/EMPIAR11526_128x128_valset.mrc'
    },
    {
        'protein': 'EMPIAR10166_128',
        'model': 'anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128',
        'val_dataset': f'{data_dir}/EMPIAR10166_128x128_valset.mrc'
    },
    {
        'protein': 'EMPIAR10786_128',
        'model': 'anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128',
        'val_dataset': f'{data_dir}/EMPIAR10786_128x128_valset.mrc'
    },
    {
        'protein': 'EMPIAR10076_256',
        'model': 'anonymousneurips008/empiar10076-ddpm-ema-cryoem-256x256',
        'val_dataset': f'{data_dir}/EMPIAR10076_256x256_valset.mrc'
    },
    {
        'protein': 'EMPIAR10648_256',
        'model': 'anonymousneurips008/empiar10648-ddpm-cryoem-256x256',
        'val_dataset': f'{data_dir}/EMPIAR10648_256x256_valset.mrc'
    },
]

# -------------------------------
# Worker Process
# -------------------------------

def run_reconstruction(start_id, end_id, args_dict, gpu_id):
    cmd = [
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'python', 'comparative_methods/baselines.py',
        '--use_cryoem',
        '--cryoem_path', args_dict['val_dataset'],
        '--result_dir', args_dict['result_dir'],
        '--block_size', str(args_dict['block_size']),
        '--num_masks', str(args_dict['num_masks']),
        '--mask_type', args_dict['mask_type'],
        '--baseline', args_dict['baseline'],
        '--lambda_', str(args_dict['lambda']),
        '--learning_rate', str(args_dict['lr']),
        '--batch_size', '16',
        '--start_id', str(start_id),
        '--end_id', str(end_id),
        '--model', args_dict['model']
    ]

    log_file = os.path.join(args_dict['result_dir'], f"log_{start_id}_{end_id}.txt")
    full_cmd = ' '.join(cmd) + f" > {log_file} 2>&1"
    process = subprocess.Popen(full_cmd, shell=True)
    process.wait()

    if process.returncode == 0:
        print(f"[GPU {gpu_id}] Chunk {start_id}-{end_id} completed successfully")
    else:
        print(f"[GPU {gpu_id}] Chunk {start_id}-{end_id} failed with return code {process.returncode}")


if __name__ == "__main__":
    for experiment in experiments:
        protein = experiment['protein']
        val_dataset = experiment['val_dataset']
        model_path = experiment['model']

        if val_dataset.endswith('.mrcs') or val_dataset.endswith('.mrc'):
            import mrcfile
            total_images = torch.from_numpy(mrcfile.open(val_dataset).data.copy()).shape[0]
        else:
            total_images = torch.load(val_dataset).shape[0]

        for config in configs:
            block_size = config['block_size']
            num_masks = config['num_masks']
            mask_type = config['mask_type']

            result_dir = os.path.join('results_3d', protein, baseline, f"block_{block_size}_masks_{num_masks}_{mask_type}")
            results_2d_dir = os.path.join('results', protein, baseline, f"block_{block_size}_masks_{num_masks}_{mask_type}")
            os.makedirs(result_dir, exist_ok=True)

            if baseline == 'dmplug':
                lr, lmda = 0, 0
                mask_type = 'superres'
            else:
                best_params_path = os.path.join(results_2d_dir, 'best_params.json')
                if not os.path.exists(best_params_path):
                    raise ValueError(f"Missing best_params.json in {results_2d_dir}")
                with open(best_params_path, 'r') as f:
                    best = json.load(f)
                lr = best['lr']
                lmda = best['lambda']

            completed_ids = [
                int(f.split('_')[-1].replace('.pt', ''))
                for f in os.listdir(result_dir)
                if f.startswith('reconstruction_raw_image_') and f.endswith('.pt') and f.split('_')[-1].replace('.pt', '').isdigit()
            ]
            resume_start = max(completed_ids) + 1 if completed_ids else 0

            print(f"[{protein}] Resuming from image {resume_start} of {total_images}")

            chunks = [
                (start, min(start + chunk_size - 1, total_images - 1))
                for start in range(resume_start, total_images, chunk_size)
            ]

            i = 0
            while i < len(chunks):
                processes = []
                for j in range(num_parallel):
                    if i + j >= len(chunks):
                        break
                    start_id, end_id = chunks[i + j]
                    args_dict = {
                        'val_dataset': val_dataset,
                        'result_dir': result_dir,
                        'block_size': block_size,
                        'num_masks': num_masks,
                        'mask_type': mask_type,
                        'baseline': baseline,
                        'lambda': lmda,
                        'lr': lr,
                        'model': model_path
                    }
                    gpu_id = gpu_ids[(i + j) % len(gpu_ids)]
                    p = Process(target=run_reconstruction, args=(start_id, end_id, args_dict, gpu_id))
                    processes.append(p)
                    p.start()
                    time.sleep(1)

                for p in processes:
                    p.join()

                i += num_parallel