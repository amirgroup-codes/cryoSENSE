"""
Core implementation of the CryoGEN algorithm.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import gc

def measurement_operator(image, masks, block_size):
    """
    Apply the measurement operator efficiently using broadcasting and pooling.

    Args:
        image: Tensor of shape [batch_size, channels, img_size, img_size] in [-1, 1] range
        masks: Tensor of shape [num_masks, img_size, img_size]
        block_size: Block size for downsampling
    
    Returns:
        List of measurements, each of shape [batch_size, channels, output_size, output_size]
    """
    batch_size, channels, img_size, _ = image.shape
    num_masks = masks.shape[0]

    # 1. Prepare tensors for broadcasting
    # image: [B, C, H, W] -> [B, 1, C, H, W]
    # masks: [M, H, W] -> [1, M, 1, H, W]
    img_expanded = image.unsqueeze(1)
    masks_expanded = masks.view(1, num_masks, 1, img_size, img_size)

    # 2. Perform element-wise multiplication using broadcasting
    # Result shape: [B, M, C, H, W]
    masked_images = img_expanded * masks_expanded

    # 3. Reshape for batch processing with pooling
    # [B, M, C, H, W] -> [B * M, C, H, W]
    masked_images_flat = masked_images.view(batch_size * num_masks, channels, img_size, img_size)

    # 4. Apply block-wise summation using pooling
    # Result shape: [B * M, C, H // block, W // block]
    measurements_flat = block_wise_sum_pooling(masked_images_flat, block_size)

    # 5. Reshape back to separate batch and mask dimensions
    output_h = img_size // block_size
    output_w = img_size // block_size
    # Result shape: [B, M, C, H_out, W_out]
    measurements_batch = measurements_flat.view(batch_size, num_masks, channels, output_h, output_w)

    # 6. Convert back to a list to match original function's output format
    measurements_list = [measurements_batch[:, i] for i in range(num_masks)]
    
    return measurements_list

def block_wise_sum_pooling(image, block_size):
    """
    Efficiently calculates the sum within non-overlapping blocks using pooling.

    Args:
        image: Tensor of shape [Batch, Channels, Height, Width].
        block_size: The size of the square blocks.

    Returns:
        Tensor of shape [Batch, Channels, Height // block_size, Width // block_size]
        containing the sum within each block.
    """
    # Handle empty tensor case if masking results in empty blocks
    if image.numel() == 0:  
        output_h = image.shape[2] // block_size
        output_w = image.shape[3] // block_size
        return torch.zeros(image.shape[0], image.shape[1], output_h, output_w,
                           dtype=image.dtype, device=image.device)

    # avg_pool2d calculates sum / N. divisor_override=1 makes it calculate sum / 1.
    # This directly computes the sum within each block.
    return F.avg_pool2d(image, kernel_size=block_size, stride=block_size, divisor_override=1)

def measurement_consistency_gradient(current_x0_estimate, target_measurements, masks, block_size):
    """
    Compute the gradient of ||y - A(x0)||^2 with respect to x0.
    
    Args:
        current_x0_estimate: Current estimate of x0 [batch_size, channels, img_size, img_size] in [-1, 1] range
        target_measurements: List of target measurements, each element is a tensor of shape [batch_size, channels, output_size, output_size]
        masks: Binary masks
        block_size: Block size for downsampling
    
    Returns:
        Gradient tensor of shape [batch_size, channels, img_size, img_size]
    """
    # Create a copy with requires_grad=True
    x0 = current_x0_estimate.clone().detach().requires_grad_(True)
    
    # Apply forward measurement operator
    predicted_measurements = measurement_operator(x0, masks, block_size)
    
    # Compute measurement consistency loss across all masks and all images in batch
    # Stack measurements into shape [num_masks, batch_size, channels, output_size, output_size]
    predicted_stacked = torch.stack(predicted_measurements)  # [num_masks, batch_size, channels, output_size, output_size]
    target_stacked = torch.stack(target_measurements)  # [num_masks, batch_size, channels, output_size, output_size]
    
    # Reshape to align masks as a batch dimension for more efficient computation
    # Move from [num_masks, batch_size, channels, output_size, output_size] to 
    # [batch_size, num_masks, channels, output_size, output_size]
    predicted_stacked = predicted_stacked.permute(1, 0, 2, 3, 4)
    target_stacked = target_stacked.permute(1, 0, 2, 3, 4)
    
    # Calculate element-wise squared error with efficient broadcasting
    # This computes all errors in a single operation
    loss = F.mse_loss(predicted_stacked, target_stacked)

    # Compute gradient
    loss.backward()
    
    return x0.grad/(torch.norm(x0.grad) + 1e-8)

def cryogen_sampling(target_measurements_batch, masks, batch_size, unet, scheduler, block_size,
                    num_timesteps=1000, zeta_scale=1e-1, zeta_min=1e-2, beta=0.9, beta_min=0.1, 
                    device="cuda", callback=None):
    """
    Implement the CryoGEN algorithm using DDPM sampling with momentum.
    
    Args:
        target_measurements_batch: List of target measurements from original images
        masks: Binary masks used for measurements
        batch_size: Number of images in the batch
        unet: UNet model from the pipeline
        scheduler: DDPM scheduler from the pipeline
        block_size: Block size for downsampling
        num_timesteps: Number of diffusion timesteps
        zeta_scale: Final scale factor for the step size (default: 1e-1)
        zeta_min: Initial scale factor for the step size (default: 1e-2)
        beta: Final momentum factor (default: 0.9)
        beta_min: Initial momentum factor (default: 0.1)
        device: Computation device ("cuda" or "cpu")
        callback: Optional callback function for monitoring progress
        
    Returns:
        Reconstructed images in [-1, 1] range with shape [batch_size, channels, img_size, img_size]
    """
    # Get image dimensions from the model config
    img_size = unet.config.sample_size
    in_channels = unet.config.in_channels
    
    # Initialize from Gaussian noise
    x_T = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
    x_t = x_T.clone()
    
    # Initialize momentum buffer
    momentum = torch.zeros_like(x_T)
    
    # Set the number of inference steps
    scheduler.set_timesteps(num_timesteps)
    
    # Get DDPM timesteps (usually high to low)
    timesteps = scheduler.timesteps
    pbar = tqdm(timesteps)
    
    # Iterate using DDPM timesteps
    for i, t in enumerate(pbar):
        # Calculate current beta value using linear schedule
        current_beta = beta_min + (beta - beta_min) * (i / (len(timesteps) - 1))
        
        # Calculate current zeta value using linear schedule
        current_zeta = zeta_min + (zeta_scale - zeta_min) * (i / (len(timesteps) - 1))
        
        # Calculate step_idx for scheduler
        step_idx = (pbar.total - 1) - pbar.n
        
        # Ensure t is a tensor for model input
        timestep_tensor = t if isinstance(t, torch.Tensor) else torch.tensor([t], device=device)
        
        # Compute noise prediction
        with torch.no_grad():
            noise_pred = unet(x_t, timestep_tensor).sample
        
        # Use the DDPM scheduler step function
        step_output = scheduler.step(noise_pred, t, x_t)
        x_prev_ddpm = step_output.prev_sample  # DDPM estimate for x_{t-1}
        x0_estimate_ddpm = step_output.pred_original_sample  # DDPM estimate for x0
        
        # Clamp the estimated x0
        x0_estimate_ddpm = torch.clamp(x0_estimate_ddpm, -1, 1)
        
        # Calculate lookahead x0 estimate using current_beta (Nesterov momentum)
        x0_lookahead = x0_estimate_ddpm - current_beta * momentum
        x0_lookahead = torch.clamp(x0_lookahead, -1, 1)

        # Compute gradient at the lookahead point
        grad_batch_nesterov = measurement_consistency_gradient(x0_lookahead, target_measurements_batch, masks, block_size)

        # Update momentum using the Nesterov gradient
        momentum = current_beta * momentum + current_zeta * grad_batch_nesterov

        # Apply gradient guidance using the updated momentum
        x_prev_guided = x_prev_ddpm - momentum
        
        # Update x_t for the next iteration
        x_t = x_prev_guided
        
        # Clamp to valid range
        x_t = torch.clamp(x_t, -1, 1)
        
        # Update progress info
        t_value = t.item() if isinstance(t, torch.Tensor) else t
        pbar.set_description(f"Step: {step_idx} (t={t_value}), Beta: {current_beta:.3f}, Zeta: {current_zeta:.4f}")
        
        # Call the callback function if provided
        if callback:
            callback(i, x_t)
        
        # Clean up memory every 100 steps
        if step_idx % 100 == 0:
            torch.cuda.empty_cache()
    
    return x_t 