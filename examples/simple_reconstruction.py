#!/usr/bin/env python
"""
Simple example of using CryoGEN for image reconstruction.
"""

import os
import argparse
from CryoGEN import CryoGEN

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CryoGEN Simple Example')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pretrained DDPM model')
    parser.add_argument('--cryoem_path', type=str, required=True,
                        help='Path to the CryoEM dataset file (.pt or .mrcs format)')
    parser.add_argument('--image_id', type=int, default=0,
                        help='ID of the image to reconstruct (default: 0)')
    parser.add_argument('--block_size', type=int, default=4,
                        help='Block size for downsampling (default: 4)')
    parser.add_argument('--result_dir', type=str, default='example_results',
                        help='Directory to save results (default: example_results)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose mode with detailed visualization outputs')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    print(f"CryoGEN Simple Example")
    print(f"--------------------")
    print(f"Model: {args.model}")
    print(f"CryoEM data: {args.cryoem_path}")
    print(f"Image ID: {args.image_id}")
    print(f"Block size: {args.block_size}")
    print(f"Results directory: {args.result_dir}")
    print(f"Verbose mode: {'Enabled' if args.verbose else 'Disabled'}")
    print()
    
    # Initialize CryoGEN
    cryogen = CryoGEN(
        model_path=args.model,
        block_size=args.block_size,
        result_dir=args.result_dir,
        verbose=args.verbose
    )
    
    # Reconstruct a single image with default parameters
    reconstructed_image, original_image, metrics = cryogen.reconstruct_from_cryoem(
        file_path=args.cryoem_path,
        image_ids=[args.image_id]
    )
    
    print(f"\nReconstruction Results:")
    print(f"PSNR: {metrics[0]['PSNR']:.2f} dB")
    print(f"SSIM: {metrics[0]['SSIM']:.4f}")
    print(f"MSE: {metrics[0]['MSE']:.6f}")
    print(f"\nResults saved to {args.result_dir}")
    
    if args.verbose:
        print(f"\nVerbose mode outputs:")
        print(f" - All masks saved to {args.result_dir}/masks/")
        print(f" - All measurements saved to {args.result_dir}/measurements/")
        print(f" - Diffusion process steps saved to {args.result_dir}/diffusion_process/")
        print(f" - Diffusion process animation: {args.result_dir}/diffusion_process_img{args.image_id}.gif")

if __name__ == "__main__":
    main() 