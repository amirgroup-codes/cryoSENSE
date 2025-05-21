#!/bin/bash

# Simple CryoGEN demonstration script
# Running with block size 32 and 1024 masks
# 
# Parameters:
# - Block size: 32 (automatically uses zeta_scale=10.0 from config)
# - Number of masks: 1024
# - Model: anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128
# - Data: data/sample_empiar10076.pt

# Set output directory
RESULT_DIR="results/block32_1024masks"

# Create results directory
mkdir -p $RESULT_DIR

# Run CryoGEN with the specified parameters
cryogen --model anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128 \
        --cryoem_path data/sample_empiar10076.pt \
        --block_size 32 \
        --num_masks 1024 \
        --start_id 0 \
        --end_id 0 \
        --use_config \
        --verbose \
        --result_dir $RESULT_DIR


# Running with block size 2 and 4 masks
# 
# Parameters:
# - Block size: 2 (automatically uses zeta_scale=1.0 from config)
# - Number of masks: 4
# - Model: anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128
# - Data: data/sample_empiar10076.pt

# Set output directory
RESULT_DIR="results/block2_4masks"

# Create results directory
mkdir -p $RESULT_DIR

# Run CryoGEN with the specified parameters
cryogen --model anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128 \
        --cryoem_path data/sample_empiar10076.pt \
        --block_size 2 \
        --num_masks 4 \
        --start_id 0 \
        --end_id 0 \
        --use_config \
        --verbose \
        --result_dir $RESULT_DIR

echo
echo "Process complete! Check results in $RESULT_DIR" 