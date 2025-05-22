"""
Functions for evaluating and visualizing reconstructions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from .core import measurement_operator
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_measurements_and_original(original_image, measurements, masks, result_dir="results"):
    """
    Plot and save the original image and its measurements.
    
    Args:
        original_image: Original image tensor [batch_size, channels, img_size, img_size] in [-1, 1] range
        measurements: List of measurements, each of shape [batch_size, channels, output_size, output_size]
        masks: Binary masks used for measurements
        result_dir: Directory to save results
    """
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "measurements"), exist_ok=True)
    
    # Get dimensions
    batch_size = original_image.shape[0]
    
    # Process each image in the batch
    for b in range(batch_size):
        # Get the original image
        img = original_image[b]
        
        # Convert from [-1, 1] to [0, 1] for visualization
        img_vis = (img + 1) / 2
        
        # Create a figure for the original image with no white space
        plt.figure(figsize=(10, 10), frameon=False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if img.shape[0] == 1:  # Single channel
            plt.imshow(img_vis[0].cpu().numpy(), cmap='gray')
        else:  # RGB
            plt.imshow(img_vis.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.savefig(f"{result_dir}/original_image_{b}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def analyze_reconstruction(original_image_batch, reconstructed_image_batch, target_measurements_batch, 
                           masks, block_size, current_batch_ids=None, result_dir="results", experiment_params=None):
    """
    Analyze the quality of the reconstruction for a batch of images.
    
    Args:
        original_image_batch: Original image tensor [batch_size, channels, img_size, img_size]
        reconstructed_image_batch: Reconstructed image tensor [batch_size, channels, img_size, img_size]
        target_measurements_batch: List of target measurements for the batch
        masks: Binary masks used for measurements
        block_size: Block size for downsampling 
        current_batch_ids: List of image IDs in the current batch (optional)
        result_dir: Directory to save results
        experiment_params: Dictionary of experiment parameters (optional)
        
    Returns:
        List of dictionaries containing all computed metrics for each image in the batch
    """
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize LPIPS loss function
    loss_fn_alex = lpips.LPIPS(net='alex').to(original_image_batch.device)
    
    batch_size = original_image_batch.shape[0]
    all_metrics = []
    
    # If no batch IDs provided, create sequential IDs
    if current_batch_ids is None:
        current_batch_ids = list(range(batch_size))
    
    for j in range(batch_size):
        image_id = current_batch_ids[j]
        print(f"\n----- Analyzing reconstruction for image {image_id} -----")
        
        # Extract single images from batch
        original_image = original_image_batch[j:j+1]  # Keep batch dimension for LPIPS
        reconstructed_image = reconstructed_image_batch[j:j+1]
        
        # Extract single measurements for this image from the batch
        target_measurements_single = [m_batch[j:j+1] for m_batch in target_measurements_batch]
        
        # 1. Calculate pixel-wise metrics
        mse = F.mse_loss(original_image, reconstructed_image).item()
        mae = F.l1_loss(original_image, reconstructed_image).item()
        
        # Convert images from [-1, 1] to [0, 1] range for visualization and metrics
        original_vis = (original_image + 1) / 2
        reconstructed_vis = (reconstructed_image + 1) / 2
        
        # Convert to numpy for skimage metrics
        if original_image.shape[1] > 1:  # Multi-channel
            # Take first 3 channels for RGB or just first channel for grayscale
            original_numpy = original_vis[0, :min(3, original_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
            reconstructed_numpy = reconstructed_vis[0, :min(3, reconstructed_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
            # Ensure channel axis is correct for multi-channel images
            channel_axis = 2
        else:
            original_numpy = original_vis[0, 0].cpu().numpy()
            reconstructed_numpy = reconstructed_vis[0, 0].cpu().numpy()
            # For grayscale images
            channel_axis = None
        
        # Calculate PSNR and SSIM using skimage implementations
        psnr = peak_signal_noise_ratio(original_numpy, reconstructed_numpy, data_range=1.0)
        
        # Calculate SSIM with appropriate parameters
        if channel_axis is not None:
            ssim = structural_similarity(original_numpy, reconstructed_numpy, channel_axis=channel_axis, data_range=1.0)
        else:
            ssim = structural_similarity(original_numpy, reconstructed_numpy, data_range=1.0)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_value = loss_fn_alex(original_image, reconstructed_image).item()
        
        # Calculate the L2 norm distance between original and reconstructed images
        l2_norm = torch.norm(original_image - reconstructed_image, p=2).item()
        
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
        print(f"LPIPS: {lpips_value:.4f}")
        print(f"L2 Norm Distance: {l2_norm:.6f}")
        
        # 2. Calculate measurement-wise error
        reconstructed_measurements = measurement_operator(reconstructed_image, masks, block_size)
        
        measurement_mses = []
        for i in range(len(target_measurements_single)):
            mse_i = F.mse_loss(reconstructed_measurements[i], target_measurements_single[i]).item()
            measurement_mses.append(mse_i)
        
        avg_measurement_mse = sum(measurement_mses) / len(measurement_mses)
        print(f"Average Measurement MSE: {avg_measurement_mse:.6f}")
        
        # 3. Visualize results
       

        fig = plt.figure(
            figsize=(10, 3),   # ↓ shorter / narrower canvas
            dpi=300,
            constrained_layout=False  # we'll tune layout by hand
        )

        gs = gridspec.GridSpec(
            1, 4,
            width_ratios=[1, 1, 1, 0.04],   # even slimmer colour-bar track
            wspace=0.01                     # virtually no gap between images
        )

        # Axes
        ax_orig  = fig.add_subplot(gs[0, 0])
        ax_recon = fig.add_subplot(gs[0, 1])
        ax_err   = fig.add_subplot(gs[0, 2])
        cax      = fig.add_subplot(gs[0, 3])       # colour-bar axis

        # --------------- (plotting code unchanged) ---------------------------
        ax_orig.imshow(original_vis.squeeze().cpu().numpy(), cmap='gray')
        ax_orig.set_title("Original", fontsize=10);  ax_orig.axis("off")

        ax_recon.imshow(reconstructed_vis.squeeze().cpu().numpy(), cmap='gray')
        ax_recon.set_title("Reconstructed", fontsize=10);  ax_recon.axis("off")

        err = torch.abs(original_image - reconstructed_image).squeeze().cpu().numpy()
        im_err = ax_err.imshow(err, cmap='hot')
        ax_err.set_title("Error Map", fontsize=10);  ax_err.axis("off")

        fig.colorbar(im_err, cax=cax, label="Absolute\nError")

        # ---------------- fine-tune outer margins ----------------------------
        fig.subplots_adjust(
            left=0.01,   # tighten the frame
            right=0.99,
            top=0.98,
            bottom=0.02
        )

        plt.savefig(
            f"{result_dir}/reconstruction_comparison_image_{image_id}.png",
            bbox_inches="tight",   # trims remaining border
            pad_inches=0.02        # *tiny* padding so titles aren’t clipped
        )
        plt.close()
        
        # 4. Save metrics to CSV
        metrics = {
            'image_id': image_id,
            'block_size': block_size,
            'num_masks': len(masks),
            'mask_type': "NA",  # Default value, overridden by experiment_params if provided
            'noise_level': 0.0,  # Default value, overridden by experiment_params if provided
            'MSE': mse,
            'MAE': mae,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips_value,
            'L2_norm': l2_norm,
            'Avg_measurement_MSE': avg_measurement_mse
        }
        
        # Update with experiment parameters if provided
        if experiment_params:
            metrics.update(experiment_params)
        
        all_metrics.append(metrics)
        save_metrics_to_csv(metrics, result_dir)
    
    return all_metrics

def save_metrics_to_csv(metrics, result_dir):
    """
    Save reconstruction metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics (MSE, MAE, PSNR, SSIM, LPIPS, L2_norm, Avg_measurement_MSE)
        result_dir: Directory to save the CSV file
    """
    csv_path = os.path.join(result_dir, "reconstruction_metrics.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_id', 'block_size', 'num_masks', 'mask_type', 'noise_level', 'MSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'L2_norm', 'Avg_measurement_MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)
    
    print(f"Metrics saved to {csv_path}")
    
    # Also save a summary text file for easy reference
    with open(os.path.join(result_dir, f"metrics_summary_image_{metrics['image_id']}.txt"), 'w') as f:
        f.write(f"----- Reconstruction Quality for Image {metrics['image_id']} -----\n")
        f.write(f"MSE: {metrics['MSE']:.6f}\n")
        f.write(f"MAE: {metrics['MAE']:.6f}\n")
        f.write(f"PSNR: {metrics['PSNR']:.2f} dB\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"LPIPS: {metrics['LPIPS']:.4f}\n")
        f.write(f"L2 Norm Distance: {metrics['L2_norm']:.6f}\n")
        f.write(f"Average Measurement MSE: {metrics['Avg_measurement_MSE']:.6f}\n")
        f.write(f"Noise Level (sigma): {metrics.get('noise_level', 0.0):.6f}\n") 