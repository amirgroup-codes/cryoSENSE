#!/bin/bash
# Script to run multiple CryoGEN reconstruction jobs with different parameters

# Common parameters
MODEL="anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128"
DATASET="data/sample_empiar10076.pt"
RESULTS_BASE="results/reconstructions"
GPU_IDS="0,1,2,3"

# Create base results directory
mkdir -p "$RESULTS_BASE"

echo "===== Starting CryoGEN Batch Reconstructions ====="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "GPUs: $GPU_IDS"

# Run block size 2 with 1 masks
echo "Running reconstruction: Block size 2, 1 masks..."
python3 scripts/reconstruct_all_images.py \
  --result_dir "${RESULTS_BASE}/bs32_masks1024" \
  --batch_size 256 \
  --block_size 2 \
  --num_masks 1 \
  --mask_type "random_binary" \
  --gpu_ids "$GPU_IDS" \
  --use_config \
  --single_dataset \
  --protein_name "empiar10076_128" \
  --model "$MODEL" \
  --dataset "$DATASET"
echo "Completed block size 32 with 1024 masks"

echo "===== All reconstructions completed ====="
echo "Results saved in $RESULTS_BASE"

# Optionally, add a comparison summary
echo "===== Quality Comparison ====="
for DIR in "${RESULTS_BASE}"/bs*_masks*; do
  if [ -d "$DIR" ]; then
    METRICS="${DIR}/empiar10076_128/reconstruction_summary.json"
    if [ -f "$METRICS" ]; then
      CONFIG=$(basename "$DIR")
      echo "$CONFIG:"
      grep -E "PSNR|SSIM|LPIPS" "$METRICS" | sed 's/[{}":]//g' | sed 's/,$//' | sed 's/^/  /'
    fi
  fi
done 