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
        mask_type: Type of mask - "random_binary", "random_gaussian", "checkerboard", or "moire"
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
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. Choose 'random_binary', 'random_gaussian', 'checkerboard', or 'moire'.")
    
    return masks 