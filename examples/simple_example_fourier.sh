#!/bin/bash

# CryoGEN Fourier Masking Demonstration Script
#
# This script demonstrates random_fourier mask-based 3D reconstruction on EMPIAR-10076
# Experiments are run with different mask probabilities to evaluate reconstruction quality
# at various levels of Fourier domain subsampling.
#
# Parameters:
# - Model: anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128
# - Data: data/sample_empiar10076.pt
# - Block size: 1 (no downsampling, full resolution)
# - Number of masks: 1
# - Mask type: random_fourier (Fourier domain subsampling)
# - Mask probabilities: 0.1, 0.2, 0.3, 0.4, 0.5
#
# Each mask_prob value represents the probability that a Fourier coefficient is sampled:
#   - 0.1 = 10% of Fourier coefficients (high compression)
#   - 0.2 = 20% of Fourier coefficients
#   - 0.3 = 30% of Fourier coefficients
#   - 0.4 = 40% of Fourier coefficients
#   - 0.5 = 50% of Fourier coefficients (moderate compression)

echo "=========================================="
echo "CryoGEN Fourier Masking Experiments"
echo "EMPIAR-10076 Dataset"
echo "=========================================="
echo ""

# Array of mask probabilities to test
MASK_PROBS=(0.5)

# Loop through each mask probability
for MASK_PROB in "${MASK_PROBS[@]}"
do
    echo "------------------------------------------"
    echo "Running experiment with mask_prob = $MASK_PROB"
    echo "------------------------------------------"

    # Set output directory based on mask probability
    RESULT_DIR="results/fourier_block1_maskprob${MASK_PROB}"

    # Create results directory
    mkdir -p $RESULT_DIR

    # Run CryoGEN with random_fourier masking
    cryogen --model anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128 \
            --cryoem_path data/sample_empiar10076.pt \
            --block_size 1 \
            --num_masks 1 \
            --mask_type random_fourier \
            --mask_prob $MASK_PROB \
            --start_id 0 \
            --end_id 0 \
            --use_config \
            --verbose \
            --result_dir $RESULT_DIR

    echo ""
    echo "Completed mask_prob = $MASK_PROB"
    echo "Results saved to: $RESULT_DIR"
    echo ""
done

echo "=========================================="
echo "All Fourier masking experiments complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for MASK_PROB in "${MASK_PROBS[@]}"
do
    RESULT_DIR="results/fourier_block1_maskprob${MASK_PROB}"
    echo "  - mask_prob $MASK_PROB: $RESULT_DIR"
done
echo ""
echo "Compare reconstruction quality across different Fourier sampling rates."
echo "Expected trend: Higher mask_prob → Better reconstruction quality"
