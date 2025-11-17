"""
Utilities for loading and processing CryoEM data.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import mrcfile

def load_cryoem_image(file_path, image_id=0, device="cuda"):
    """
    Load a specific image from a CryoEM dataset stored in PyTorch format or mrcfile format.
    
    Args:
        file_path: Path to the .pt or .mrcs file containing CryoEM images
        image_id: Index of the image to load (default: 0)
        device: Device to place tensor on
        
    Returns:
        Selected image tensor in the dataset with shape [1, 1, img_size, img_size]
    """
    print(f"Loading CryoEM data from {file_path}...")
    
    # Load the file based on extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pt':
            # Load PyTorch tensor file
            cryoem_data = torch.load(file_path, map_location=device)
            print(f"CryoEM data loaded successfully from .pt file. Data shape: {cryoem_data.shape}")
        elif file_ext == '.mrcs':
            # Load MRC file using mrcfile library
            with mrcfile.open(file_path) as mrc:
                cryoem_data = torch.from_numpy(mrc.data)
            print(f"CryoEM data loaded successfully from .mrcs file. Data shape: {cryoem_data.shape}")
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Supported formats are .pt and .mrcs")
        
        # Check if the requested image_id is valid
        if image_id >= cryoem_data.shape[0]:
            print(f"Warning: Requested image_id {image_id} exceeds dataset size {cryoem_data.shape[0]}. Using image 0 instead.")
            image_id = 0
            
        # Extract the selected image
        selected_image = cryoem_data[image_id]
        print(f"Selected image index: {image_id}")
        
        # Check dimensions and reshape if needed
        if selected_image.dim() == 2:  # If image is [height, width]
            selected_image = selected_image.unsqueeze(0)  # Convert to [1, height, width]
            
        if selected_image.dim() == 3 and selected_image.shape[0] > 1:  # If image is [batch, height, width]
            selected_image = selected_image[0].unsqueeze(0)  # Convert to [1, height, width]
            
        print(f"Selected image shape: {selected_image.shape}")
        
        # Normalize to [-1, 1] range
        if selected_image.min() >= 0 and selected_image.max() <= 1:
            # If in [0, 1] range, convert to [-1, 1]
            selected_image = selected_image * 2 - 1
            print("Image normalized from [0, 1] to [-1, 1] range")
        elif selected_image.min() >= 0 and selected_image.max() <= 255:
            # If in [0, 255] range, convert to [-1, 1]
            selected_image = selected_image / 127.5 - 1
            print("Image normalized from [0, 255] to [-1, 1] range")
        else:
            # Otherwise, normalize to [-1, 1] range
            selected_image = (selected_image - selected_image.min()) / (selected_image.max() - selected_image.min() + 1e-8)
            selected_image = selected_image * 2 - 1
            print("Image normalized to [-1, 1] range")
            
        return selected_image.unsqueeze(0).to(device)  # Add batch dimension [1, channels, H, W]
        
    except Exception as e:
        print(f"Error loading CryoEM data: {e}")
        raise

def load_cryoem_batch(file_path, image_ids, target_img_size, device="cuda"):
    """
    Load a batch of images from a CryoEM dataset stored in PyTorch or mrcfile format.
    
    Args:
        file_path: Path to the .pt or .mrcs file containing CryoEM images
        image_ids: List of indices of the images to load
        target_img_size: Target image size for resizing if necessary
        device: Device to place tensor on
        
    Returns:
        Batch tensor of images with shape [batch_size, 1, img_size, img_size]
    """
    print(f"Loading batch of CryoEM data from {file_path}...")
    
    # Determine file type from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Load the file based on extension
    try:
        # Load to CPU first for safety with large datasets
        if file_ext == '.pt':
            cryoem_data = torch.load(file_path, map_location='cpu')
            print(f"CryoEM data loaded successfully from .pt file. Data shape: {cryoem_data.shape}")
        elif file_ext == '.mrcs':
            # Load MRC file using mrcfile library
            with mrcfile.open(file_path) as mrc:
                cryoem_data = torch.from_numpy(mrc.data).to('cpu')
            print(f"CryoEM data loaded successfully from .mrcs file. Data shape: {cryoem_data.shape}")
        elif file_ext == '.mrc':
            # Load MRC file using mrcfile library
            with mrcfile.open(file_path) as mrc:
                cryoem_data = torch.from_numpy(mrc.data).to('cpu')
            print(f"CryoEM data loaded successfully from .mrc file. Data shape: {cryoem_data.shape}")
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Supported formats are .pt and .mrcs")
        
        # Check if all requested image_ids are valid
        max_image_id = max(image_ids)
        if max_image_id >= cryoem_data.shape[0]:
            print(f"Warning: Some requested image_ids exceed dataset size {cryoem_data.shape[0]}. Clamping to available range.")
            image_ids = [min(i, cryoem_data.shape[0] - 1) for i in image_ids]
            
        # Extract the selected images
        batch_data = cryoem_data[image_ids]
        print(f"Selected batch size: {len(image_ids)}")
        print(f"Selected batch shape: {batch_data.shape}")
        
        # Check dimensions and reshape if needed
        if batch_data.dim() == 2:  # If image is [batch, pixels]
            batch_data = batch_data.unsqueeze(1)  # Convert to [batch, 1, pixels]
        
        if batch_data.dim() == 3:  # If image is [batch, height, width]
            batch_data = batch_data.unsqueeze(1)  # Convert to [batch, 1, height, width]
            
        # Normalize to [-1, 1] range
        if batch_data.min() >= 0 and batch_data.max() <= 1:
            # If in [0, 1] range, convert to [-1, 1]
            batch_data = batch_data * 2 - 1
            print("Batch normalized from [0, 1] to [-1, 1] range")
        elif batch_data.min() >= 0 and batch_data.max() <= 255:
            # If in [0, 255] range, convert to [-1, 1]
            batch_data = batch_data / 127.5 - 1
            print("Batch normalized from [0, 255] to [-1, 1] range")
        else:
            # Otherwise, normalize to [-1, 1] range
            # Apply min-max normalization to the entire batch
            min_val = batch_data.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            max_val = batch_data.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            batch_data = (batch_data - min_val) / (max_val - min_val + 1e-8)
            batch_data = batch_data * 2 - 1
            print("Batch normalized to [-1, 1] range")
            
        # Resize if necessary to match the expected model input size
        if batch_data.shape[2] != target_img_size or batch_data.shape[3] != target_img_size:
            print(f"Resizing batch from {batch_data.shape[2]}x{batch_data.shape[3]} to {target_img_size}x{target_img_size}")
            batch_data = F.interpolate(
                batch_data, 
                size=(target_img_size, target_img_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Move batch to device
        batch_data = batch_data.to(device)
        
        # Free up memory
        del cryoem_data
        torch.cuda.empty_cache()
        
        return batch_data  # Shape: [batch_size, channels, target_img_size, target_img_size]
        
    except Exception as e:
        print(f"Error loading CryoEM data batch: {e}")
        raise

def add_gaussian_noise(measurements, noise_level):
    """
    Add Additive White Gaussian Noise (AWGN) to measurements.
    
    Args:
        measurements: List of measurements, each of shape [batch_size, channels, output_size, output_size]
        noise_level: Standard deviation of the noise (sigma)
    
    Returns:
        List of noisy measurements with the same shape as input
    """
    if noise_level <= 0:
        return measurements  # No noise to add
    
    noisy_measurements = []
    for m in measurements:
        # Generate Gaussian noise with same shape as measurement
        noise = torch.randn_like(m) * noise_level
        # Add noise to measurement
        noisy_m = m + noise
        noisy_measurements.append(noisy_m)
    
    return noisy_measurements 