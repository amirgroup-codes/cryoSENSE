#!/bin/bash

# Run the training script with xformers disabled and using wandb for logging
accelerate launch --mixed_precision="bf16" --gpu_ids="0,1,2,3,4,5,6,7" --num_processes=8 --multi_gpu --main_process_port 29507 train_unconditional.py \
  --custom_dataset_path="/usr/scratch/CryoEM/CryoSensing/empiar10648/particles_train_bicubic_256_normalized.mrcs" \
  --output_dir="ddpm-ema-cryoem-256-empiar10648-may3" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=2 \
  --learning_rate=2e-5 \
  --lr_warmup_steps=500 \
  --resolution=256 \
  --mixed_precision="bf16" \
  --logger="wandb" \
  --use_ema \
  --resume_from_checkpoint="latest" \
  --save_images_epochs=10 \