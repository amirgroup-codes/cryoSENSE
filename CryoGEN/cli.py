"""
Command-line interface for CryoGEN.
"""

import argparse
import os
import torch
from .main import CryoGEN

def main():
    """
    Main entry point for CryoGEN command-line interface.
    """
    parser = argparse.ArgumentParser(description='CryoGEN: CryoEM Image Reconstruction with Diffusion Models')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pretrained DDPM model')
    parser.add_argument('--cryoem_path', type=str, required=True,
                        help='Path to the CryoEM dataset file (.pt or .mrcs format)')
    
    # Optional arguments with sensible defaults
    parser.add_argument('--block_size', type=int, default=4,
                        help='Block size for downsampling (default: 4)')
    parser.add_argument('--num_masks', type=int, default=30,
                        help='Number of binary masks to use (default: 30)')
    parser.add_argument('--mask_prob', type=float, default=0.5,
                        help='Probability for binary mask generation (default: 0.5)')
    parser.add_argument('--mask_type', type=str, default="random_binary",
                        choices=["random_binary", "random_gaussian", "checkerboard", "moire"],
                        help='Type of mask to use (default: "random_binary")')
    parser.add_argument('--zeta_scale', type=float, default=1e-1,
                        help='Scale factor for the measurement consistency gradient step size (default: 1e-1)')
    parser.add_argument('--zeta_min', type=float, default=1e-2,
                        help='Initial scale factor for the measurement consistency gradient step size (default: 1e-2)')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps for the sampling process (default: 1000)')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Final momentum factor for gradient updates (default: 0.9)')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='Initial momentum factor for gradient updates (default: 0.1)')
    parser.add_argument('--start_id', type=int, default=0,
                        help='Starting index of the image range to process (default: 0)')
    parser.add_argument('--end_id', type=int, default=0,
                        help='Ending index of the image range to process (inclusive, default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of images to process in each batch (default: 1)')
    parser.add_argument('--noise_level', type=float, default=0.0,
                        help='Standard deviation of Gaussian noise to add to measurements (default: 0.0)')
    parser.add_argument('--result_dir', type=str, default="results",
                        help='Directory to save results (default: results)')
    parser.add_argument('--device', type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help='Device to use (default: cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose mode with detailed visualization outputs including all masks, measurements, and animated GIF of the diffusion process')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Initialize CryoGEN model
    cryogen = CryoGEN(
        model_path=args.model,
        block_size=args.block_size,
        device=args.device,
        result_dir=args.result_dir,
        verbose=args.verbose
    )
    
    # Determine the image IDs to process
    if args.end_id >= args.start_id:
        # Process a range of images
        total_images = args.end_id - args.start_id + 1
        print(f"Processing {total_images} images from index {args.start_id} to {args.end_id}")
        
        # Process images in batches
        for batch_start in range(args.start_id, args.end_id + 1, args.batch_size):
            batch_end = min(batch_start + args.batch_size - 1, args.end_id)
            current_batch_ids = list(range(batch_start, batch_end + 1))
            current_batch_size = len(current_batch_ids)
            
            print(f"\n===== Processing Batch: {current_batch_ids} =====")
            
            # Reconstruct images
            reconstructed_images, original_images, metrics = cryogen.reconstruct_from_cryoem(
                file_path=args.cryoem_path,
                image_ids=current_batch_ids,
                num_masks=args.num_masks,
                mask_prob=args.mask_prob,
                mask_type=args.mask_type,
                noise_level=args.noise_level,
                num_timesteps=args.num_timesteps,
                zeta_scale=args.zeta_scale,
                zeta_min=args.zeta_min,
                beta=args.beta,
                beta_min=args.beta_min
            )
            
            # Clean up memory between batches
            torch.cuda.empty_cache()
            
    else:
        print("Error: --end_id must be greater than or equal to --start_id")
        return
    
    print("\n===== Processing Complete =====")
    print(f"All results saved to: {args.result_dir}")

if __name__ == "__main__":
    main() 