#!/bin/bash

# Test script for CryoGEN

# Define paths (customize these for your environment)
MODEL_PATH="/usr/scratch/danial_stuff/FrugalCryo/Test/training/ddpm-ema-cryoem-128-EMPIAR10076-apr12/"
DATA_PATH="/usr/scratch/danial_stuff/FrugalCryo/Test/new_pipeline/combined/empiar10076_128_val_test_combined.pt"
RESULT_DIR="test_results"
VERBOSE_RESULT_DIR="test_results_verbose"

# Create results directories
mkdir -p $RESULT_DIR
mkdir -p $VERBOSE_RESULT_DIR

echo "Testing CryoGEN installation..."
echo "================================"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo

# Run a simple test without verbose mode
echo "Running CryoGEN test without verbose mode..."
cryogen --model $MODEL_PATH \
        --cryoem_path $DATA_PATH \
        --start_id 0 \
        --end_id 0 \
        --block_size 16 \
        --num_masks 10 \
        --mask_type random_binary \
        --num_timesteps 100 \
        --result_dir $RESULT_DIR

echo
echo "Test completed. Check results in $RESULT_DIR"
echo

# Run a test with verbose mode
echo "Running CryoGEN test with verbose mode..."
cryogen --model $MODEL_PATH \
        --cryoem_path $DATA_PATH \
        --start_id 0 \
        --end_id 0 \
        --block_size 16 \
        --num_masks 10 \
        --mask_type random_binary \
        --num_timesteps 100 \
        --result_dir $VERBOSE_RESULT_DIR \
        --verbose

echo
echo "Verbose test completed. Check results in $VERBOSE_RESULT_DIR"
echo
echo "Comparing the two directories:"
echo "- Without verbose: $(ls -la $RESULT_DIR | wc -l) files"
echo "- With verbose: $(ls -la $VERBOSE_RESULT_DIR | wc -l) files" 