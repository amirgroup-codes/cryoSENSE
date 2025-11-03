"""
Module for creating different types of binary masks for CryoGEN.
"""

import torch
import numpy as np
import os

def create_binary_masks(num_masks=30, mask_prob=0.5, mask_type="random_binary", img_size=128, device="cuda", base_seed=42):
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
    generator = torch.Generator(device=device)
    if base_seed is not None:
        generator.manual_seed(base_seed)
        
    if mask_type == "random_binary":
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

    elif mask_type == "superres":
        masks = torch.ones((num_masks, img_size, img_size), device=device)

    elif mask_type == "random_fourier":
        # Create binary mask in the same fashion as random_binary
        # Cast the torch array as a complex tensor for easy identification that we're working in Fourier
        masks = torch.bernoulli(torch.full((num_masks, img_size, img_size), mask_prob)).to(torch.complex64).to(device)

    elif mask_type == "fourier_ring":
        # Biased ring sampling 

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
        cutoffs.append(dist_flat_sorted[-1].item() + 1e-5)
        cutoffs_tensor = torch.tensor(cutoffs, device=device)

        # 4. Calculate a weight for each band based on its central-ness
        r_inner_all = cutoffs_tensor[:-1]
        r_outer_all = cutoffs_tensor[1:]
        r_mid_all = (r_inner_all + r_outer_all) / 2.0
        r_mid_sq_all = r_mid_all.pow(2)
        sigma = img_size / 4.0
        band_weights = torch.exp(-r_mid_sq_all / (2 * sigma**2))

        # 5. Determine how many bands to keep based on mask_prob
        k = int(round(mask_prob * num_bands))
        if k == 0 and mask_prob > 0: k = 1 # Ensure at least 1 band if prob > 0
        if k > num_bands: k = num_bands      # Cap at 100%

        # 6. Generate deterministically-random, weighted orderings for each mask using Gumbel-Max
        gumbel_noise = -torch.log(-torch.log(
            torch.rand(num_masks, num_bands, device=device, generator=generator) + 1e-9) + 1e-9
        )
        weighted_scores = gumbel_noise + torch.log(band_weights.unsqueeze(0) + 1e-9)
        sorted_band_indices_batch = torch.argsort(weighted_scores, dim=1, descending=True)

        # 7. Select the top k bands for each mask
        # Shape: [num_masks, k]
        bands_to_keep_batch = sorted_band_indices_batch[:, :k]

        # 8. Create masks vectorized
        # 8a. Create a map of [H, W] -> band_index
        # We use torch.bucketize to find which bin each pixel's distance falls into
        # cutoffs_tensor[1:-1] provides the (num_bands-1) internal boundaries
        internal_boundaries = cutoffs_tensor[1:-1]
        band_indices_map = torch.bucketize(dist, internal_boundaries)
        # band_indices_map is now [H, W] with values from 0 to num_bands-1

        # 8b. Use broadcasting to create the masks
        # [1, H, W]
        band_indices_expanded = band_indices_map.unsqueeze(0)
        # [num_masks, k, 1, 1]
        bands_to_keep_expanded = bands_to_keep_batch.view(num_masks, k, 1, 1)

        # Broadcast check: [1, H, W] == [num_masks, k, 1, 1] -> [num_masks, k, H, W]
        # .any(dim=1) checks if a pixel's band index is in *any* of the k chosen bands
        masks = (band_indices_expanded == bands_to_keep_expanded).any(dim=1)

        # Cast to complex to signal Fourier operation
        masks = masks.to(torch.complex64)

    elif mask_type == "fourier_radial":
        num_spokes = 100
        # 1. Calculate pixel angles relative to the center
        center = img_size // 2
        x_coords = torch.arange(img_size, device=device).float() - center
        y_coords = torch.arange(img_size, device=device).float() - center
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        angles = torch.atan2(yy, xx)
        
        # 2. Normalize angles to [0, 1] and map to discrete spoke indices
        angles_normalized = (angles + torch.pi) / (2 * torch.pi) # Map to [0, 1]
        angle_indices = (angles_normalized * num_spokes).floor()
        angle_indices = torch.clamp(angle_indices, 0, num_spokes - 1)
        
        # 3. Determine how many spokes to keep based on mask_prob
        k = int(round(mask_prob * num_spokes))
        if k == 0 and mask_prob > 0: k = 1 # Ensure at least 1 spoke
        if k > num_spokes: k = num_spokes

        # 4. Generate random permutations to select k spokes for each mask
        # Shape: [num_masks, num_spokes]
        spoke_perms = torch.rand(num_masks, num_spokes, device=device, generator=generator).argsort(dim=1)
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