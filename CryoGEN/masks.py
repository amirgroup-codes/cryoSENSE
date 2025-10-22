"""
Module for creating different types of binary masks for CryoGEN.
"""

import torch
import numpy as np
import os

def create_binary_masks(num_masks=30, mask_prob=0.5, mask_type="random_binary", img_size=128, device="cuda"):
    """
    Create binary masks for measurement.
    
    Args:
        num_masks: Number of masks to generate
        mask_prob: Probability of 1s in the mask
        mask_type: Type of mask - "random_binary", "random_gaussian", "checkerboard", or "random_fourier"
        img_size: Size of the image/mask
        device: Device to place tensors on
    
    Returns:
        Tensor of shape [num_masks, img_size, img_size]
    """
    if mask_type == "random_binary":
        # Original random masking approach - each pixel is independently random
        masks = torch.bernoulli(torch.full((num_masks, img_size, img_size), mask_prob)).to(device)
    
    elif mask_type == "checkerboard":
        # Checkerboard masking approach - blocks of zeros or ones
        block_size = min(4, img_size // 4)  # Use smaller blocks if image is small
        blocks_h = img_size // block_size
        blocks_w = img_size // block_size
        
        # Generate random block values (0 or 1) for each mask
        block_masks = torch.bernoulli(torch.full((num_masks, blocks_h, blocks_w), mask_prob)).to(device)
        
        # Initialize full-size masks
        masks = torch.zeros((num_masks, img_size, img_size)).to(device)
        
        # Expand each block to block_size×block_size pixels
        for h in range(blocks_h):
            for w in range(blocks_w):
                h_start, h_end = h*block_size, (h+1)*block_size
                w_start, w_end = w*block_size, (w+1)*block_size
                # Set all pixels in each block to the same value
                masks[:, h_start:h_end, w_start:w_end] = block_masks[:, h:h+1, w:w+1].expand(-1, block_size, block_size)
    
    elif mask_type == "random_gaussian":
        # Gaussian random matrices
        A = torch.randn(num_masks, img_size, img_size, device=device)
        A /= A.pow(2).sum(dim=0, keepdim=True).sqrt()   # column normalize
        masks = A

    elif mask_type == "random_fourier":
        # Create binary mask in the same fashion as random_binary
        # Cast the torch array as a complex tensor for easy identification that we're working in Fourier
        masks = torch.bernoulli(torch.full((num_masks, img_size, img_size), mask_prob)).to(torch.complex64).to(device)

    # elif mask_type == "random_fourier_circular":
    #     # Create binary mask, where the probability of 1s is higher in the center
    #     num_total_coeffs = img_size * img_size
    #     k = int(round(mask_prob * num_total_coeffs))

    #     # Create a probability map with higher probability at the center (low frequencies)
    #     center = img_size // 2
    #     x_coords = torch.arange(img_size, device=device).float() - center
    #     y_coords = torch.arange(img_size, device=device).float() - center
    #     xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    #     dist_sq = xx**2 + yy**2        
    #     sigma = img_size / 4.0
    #     weights = torch.exp(-dist_sq / (2 * sigma**2))
    #     weights_flat = weights.view(-1)
    #     expanded_weights = weights_flat.expand(num_masks, -1)
        
    #     # Draw k indices for each mask without replacement
    #     sampled_indices = torch.multinomial(expanded_weights, num_samples=k, replacement=False)
        
    #     # Create the final masks
    #     masks_flat = torch.zeros(num_masks, num_total_coeffs, device=device)
    #     masks_flat.scatter_(1, sampled_indices, 1.0)
    #     masks = masks_flat.view(num_masks, img_size, img_size).to(torch.complex64)

    elif mask_type == "fourier_ring":
        # --- IMPLEMENTATION (with biased ring sampling) ---
        
        # 1. Calculate radial distance for all pixels
        center = img_size // 2
        x_coords = torch.arange(img_size, device=device).float() - center
        y_coords = torch.arange(img_size, device=device).float() - center
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        
        # 2. Get sorted distances to find percentile cutoffs
        dist_flat_sorted, _ = torch.sort(dist.view(-1))
        num_total_coeffs = dist_flat_sorted.numel()
        
        # Define number of equal-area bands (higher is more granular)
        num_bands = 100 
        
        # 3. Determine radius cutoffs for each equal-area band
        cutoffs = [0.0]
        for i in range(1, num_bands):
            idx = int(i * (num_total_coeffs / num_bands))
            cutoffs.append(dist_flat_sorted[idx].item())
        cutoffs.append(dist_flat_sorted[-1].item() + 1e-5) # Add max radius
        
        # 4. Calculate a weight for each band based on its central-ness (Vectorized)
        # Convert cutoffs list to a tensor
        cutoffs_tensor = torch.tensor(cutoffs, device=device)
        
        # Calculate all midpoint radii at once
        r_inner_all = cutoffs_tensor[:-1]
        r_outer_all = cutoffs_tensor[1:]
        r_mid_all = (r_inner_all + r_outer_all) / 2.0
        r_mid_sq_all = r_mid_all.pow(2)
        
        # Calculate weights for all bands at once
        sigma = img_size / 4.0
        band_weights = torch.exp(-r_mid_sq_all / (2 * sigma**2))
        # band_weights now is a 1D tensor of shape [num_bands]
            
        # 5. Determine how many bands to keep based on mask_prob
        k = int(round(mask_prob * num_bands))
        if k == 0 and mask_prob > 0: k = 1 # Ensure at least 1 band if prob > 0
        if k > num_bands: k = num_bands     # Cap at 100%
            
        masks = torch.zeros(num_masks, img_size, img_size, device=device)
        
        # 6. Sample k bands for each mask, without replacement, using the weights
        # Expand weights for batch sampling: [num_bands] -> [num_masks, num_bands]
        expanded_weights = band_weights.expand(num_masks, -1)
        
        # Sample k indices for each mask. Shape: [num_masks, k]
        bands_to_keep_batch = torch.multinomial(expanded_weights, num_samples=k, replacement=False)

        # 7. Create each mask individually
        for i in range(num_masks):
            # Get the k bands for this specific mask
            bands_to_keep = bands_to_keep_batch[i]
            
            for band_idx in bands_to_keep:
                # Get inner and outer radius for the chosen band
                r_inner = cutoffs[band_idx]
                r_outer = cutoffs[band_idx + 1]
                
                # Create the ring for this band
                band_mask = (dist >= r_inner) & (dist < r_outer)
                masks[i, band_mask] = 1.0
        
        # Cast to complex to signal Fourier operation
        masks = masks.to(torch.complex64)

    elif mask_type == "fourier_radial":
        num_spokes = 100
        # --- NEW IMPLEMENTATION ---
        # 1. Calculate pixel angles relative to the center
        center = img_size // 2
        x_coords = torch.arange(img_size, device=device).float() - center
        y_coords = torch.arange(img_size, device=device).float() - center
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        # Create a grid of angles from -pi to pi
        angles = torch.atan2(yy, xx)
        
        # 2. Normalize angles to [0, 1] and map to discrete spoke indices
        angles_normalized = (angles + torch.pi) / (2 * torch.pi) # Map to [0, 1]
        angle_indices = (angles_normalized * num_spokes).floor()
        # Clamp to ensure indices are [0, num_spokes-1]
        angle_indices = torch.clamp(angle_indices, 0, num_spokes - 1)
        
        # 3. Determine how many spokes to keep based on mask_prob
        k = int(round(mask_prob * num_spokes))
        if k == 0 and mask_prob > 0: k = 1 # Ensure at least 1 spoke
        if k > num_spokes: k = num_spokes
            
        # 4. Generate random permutations to select k spokes for each mask
        # We create a batch of permutations for efficient sampling
        # Shape: [num_masks, num_spokes]
        spoke_perms = torch.rand(num_masks, num_spokes, device=device).argsort(dim=1)
        
        # Select the first k spokes from each permutation
        # Shape: [num_masks, k]
        spokes_to_keep_batch = spoke_perms[:, :k]
        
        # 5. Create masks by checking if a pixel's angle index is in the selected set
        # Unsqueeze angle_indices to broadcast against the batch of masks
        # [H, W] -> [1, H, W]
        angle_indices_expanded = angle_indices.unsqueeze(0)
        
        # Unsqueeze spokes_to_keep_batch to broadcast against the image dimensions
        # [num_masks, k] -> [num_masks, k, 1, 1]
        spokes_to_keep_expanded = spokes_to_keep_batch.view(num_masks, k, 1, 1)

        # Broadcasting check:
        # (angle_indices_expanded == spokes_to_keep_expanded)
        # [1, H, W] == [num_masks, k, 1, 1] -> Shape: [num_masks, k, H, W]
        # .any(dim=1) checks if a pixel belongs to *any* of the k chosen spokes
        # Result is a boolean mask of shape [num_masks, H, W]
        masks = (angle_indices_expanded == spokes_to_keep_expanded).any(dim=1)
        
        # 6. Cast to complex to signal Fourier operation
        masks = masks.to(torch.complex64)
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. Choose 'random_binary', 'random_gaussian', 'checkerboard', or 'random_fourier'.")
    
    return masks 