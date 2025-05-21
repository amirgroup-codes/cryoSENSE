#!/usr/bin/env python
"""
Simple CryoGEN demonstration script with block size 32 and 1024 masks.

Parameters:
- Block size: 32 (automatically uses zeta_scale=10.0 from config)
- Number of masks: 1024
- Model: anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128
- Data: data/sample_empiar10076.pt
"""

import os
from CryoGEN import CryoGEN

def main():
    # Set parameters
    model_path = "anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128"
    cryoem_path = "data/sample_empiar10076.pt"
    block_size = 32
    num_masks = 1024
    result_dir = "results/block32_1024masks"
    
    # Create results directory
    os.makedirs(result_dir, exist_ok=True)
    
    print("==== CryoGEN Demonstration =====")
    print(f"Block size: {block_size}")
    print(f"Number of masks: {num_masks}")
    print(f"Model: {model_path}")
    print(f"Data: {cryoem_path}")
    print(f"Results will be saved to: {result_dir}")
    print("===============================")
    print()
    
    # Initialize CryoGEN with block size 32
    # This will automatically use optimal parameters from configuration file
    cryogen = CryoGEN(
        model_path=model_path,
        block_size=block_size,
        result_dir=result_dir,
        use_config=True,
        verbose=True
    )
    
    # Reconstruct an image
    reconstructed_images, original_images, metrics = cryogen.reconstruct_from_cryoem(
        file_path=cryoem_path,
        image_ids=[0],  # Process first image
        num_masks=num_masks,
        # Parameters like zeta_scale will be determined from configuration
    )
    
    # Display metrics
    print("\nReconstruction Results:")
    print(f"PSNR: {metrics[0]['PSNR']:.2f} dB")
    print(f"SSIM: {metrics[0]['SSIM']:.4f}")
    print(f"MSE: {metrics[0]['MSE']:.6f}")
    print(f"\nResults saved to {result_dir}")

if __name__ == "__main__":
    main() 