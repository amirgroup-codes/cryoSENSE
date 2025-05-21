import math
from diffusers import DDPMPipeline, DDPMScheduler
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import gc
import time
import random
from torch.cuda.amp import autocast, GradScaler
import csv
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.ndimage import zoom

# Set fixed seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable GPU optimizations if CUDA is available
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    # Set to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    # Set to use benchmark for optimal performance
    torch.backends.cudnn.benchmark = True

# Create output directories
os.makedirs("results", exist_ok=True)

# Base configuration parameters
config = {
    "block_size": 4,  # Block size for downsampling
    "result_dir": "results"  # Default results directory
}

# Generate a sample image
def generate_sample_image(seed=42, result_dir="results", pipeline=None):
    """Generate a sample image from the DDPM using a specific seed."""
    # Set the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Generate an image using the pipeline with the specified generator
    with torch.no_grad():
        output = pipeline(generator=generator)
    
    image = output.images[0]
    os.makedirs(result_dir, exist_ok=True)
    image.save(f'{result_dir}/original_image.png')
    
    # Convert PIL Image to tensor for further processing
    image_tensor = torch.tensor(np.array(image)).float() / 255.0
    
    # Reshape to match expected dimensions
    if image_tensor.dim() == 2:  # Grayscale image
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
    else:  # RGB image
        image_tensor = image_tensor.permute(2, 0, 1)  # [channels, H, W]
    
    # Normalize to [-1, 1] range
    image_tensor = image_tensor * 2 - 1
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Shape: [1, channels, img_size, img_size]
    
    return image_tensor

# New function to load CryoEM data
def load_cryoem_image(file_path, image_id=0, result_dir="results"):
    """
    Load a specific image from a CryoEM dataset stored in PyTorch format.
    
    Args:
        file_path: Path to the .pt file containing CryoEM images
        image_id: Index of the image to load (default: 0)
        result_dir: Directory to save results
        
    Returns:
        Selected image tensor in the dataset with shape [1, 1, img_size, img_size]
    """
    print(f"Loading CryoEM data from {file_path}...")
    
    # Load the PyTorch tensor file
    try:
        cryoem_data = torch.load(file_path, map_location=device)
        print(f"CryoEM data loaded successfully. Data shape: {cryoem_data.shape}")
        
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
            
        # Save a copy of the original image (in a displayable format)
        img_for_display = ((selected_image + 1) / 2).clamp(0, 1)  # Convert back to [0, 1] for display
        img_np = img_for_display.cpu().numpy()
        
        if selected_image.shape[0] == 1:  # Single channel
            img_pil = Image.fromarray((img_np[0] * 255).astype(np.uint8), mode='L')
        else:  # Multi-channel
            img_pil = Image.fromarray((np.clip(img_np.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
        
        os.makedirs(result_dir, exist_ok=True)
        # img_pil.save(f'{result_dir}/original_cryoem_image.png')
        # print(f"Original CryoEM image saved to {result_dir}/original_cryoem_image.png")
        
        # Resize if necessary to match the expected model input size
        if selected_image.shape[1] != config["img_size"] or selected_image.shape[2] != config["img_size"]:
            print(f"Resizing image from {selected_image.shape[1]}x{selected_image.shape[2]} to {config['img_size']}x{config['img_size']}")
            selected_image = F.interpolate(
                selected_image.unsqueeze(0), 
                size=(config["img_size"], config["img_size"]), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        return selected_image.unsqueeze(0).to(device)  # Add batch dimension [1, channels, H, W]
        
    except Exception as e:
        print(f"Error loading CryoEM data: {e}")
        raise

# New function to load a batch of CryoEM images
def load_cryoem_batch(file_path, image_ids, target_img_size, device, result_dir="results"):
    """
    Load a batch of images from a CryoEM dataset stored in PyTorch format.
    
    Args:
        file_path: Path to the .pt file containing CryoEM images
        image_ids: List of indices of the images to load
        target_img_size: Target image size for resizing if necessary
        device: Torch device to move the tensor to
        result_dir: Directory to save results
        
    Returns:
        Batch tensor of images with shape [batch_size, 1, img_size, img_size]
    """
    print(f"Loading batch of CryoEM data from {file_path}...")

    try:
        if file_path.endswith('.mrcs'):
            import mrcfile
            cryoem_data = torch.from_numpy(mrcfile.open(file_path).data.copy())
        else:
            cryoem_data = torch.load(file_path, map_location='cpu')
        print(f"CryoEM data loaded successfully. Data shape: {cryoem_data.shape}")
        
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
            
        # Save copies of the original images for visualization
        os.makedirs(result_dir, exist_ok=True)
        for i, image_id in enumerate(image_ids):
            # Convert back to [0, 1] for display
            img_for_display = ((batch_data[i] + 1) / 2).clamp(0, 1)  
            img_np = img_for_display.cpu().numpy()
            
            if batch_data.shape[1] == 1:  # Single channel
                img_pil = Image.fromarray((img_np[0] * 255).astype(np.uint8), mode='L')
            else:  # Multi-channel
                img_pil = Image.fromarray((np.clip(img_np.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
            
            # img_pil.save(f'{result_dir}/original_cryoem_image_{image_id}.png')
            # print(f"Original CryoEM image {image_id} saved to {result_dir}/original_cryoem_image_{image_id}.png")
        
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
        gc.collect()
        
        return batch_data  # Shape: [batch_size, channels, target_img_size, target_img_size]
        
    except Exception as e:
        print(f"Error loading CryoEM data batch: {e}")
        raise

# Create binary masks
def create_binary_masks(num_masks=30, mask_prob=0.5, mask_type="random_binary", result_dir="results"):
    """
    Create binary masks for measurement.
    
    Args:
        num_masks: Number of masks to generate
        mask_prob: Probability of 1s in the mask
        mask_type: Type of mask - "random_binary", "random_gaussian", "checkerboard", or "moire", or "superres
        result_dir: Directory to save results
    
    Returns:
        Tensor of shape [num_masks, img_size, img_size]
    """
    img_size = config["img_size"]
    
    if mask_type == "random_binary":
        # Original random masking approach - each pixel is independently random
        masks = torch.bernoulli(torch.full((num_masks, img_size, img_size), mask_prob)).to(device)
    
    elif mask_type == "checkerboard":
        # Checkerboard masking approach - 4x4 blocks of zeros or ones
        # Calculate number of blocks in each dimension
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
    
    elif mask_type == "moire":
        # Use latticegen to create moiré patterns
        from latticegen.latticegeneration import hexlattice_gen
        import numpy as np
        
        masks = torch.zeros((num_masks, img_size, img_size)).to(device)
        
        for i in range(num_masks):
            # Create two hexagonal lattices with different rotations to generate moiré pattern
            angle1 = (i * 10) % 60  # Vary angle between masks (in degrees)
            angle2 = (angle1 + 5 + i % 10) % 60  # Slightly different angle
            
            # Scale factors control the frequency of the pattern
            scale1 = 8 + (i % 5)
            scale2 = 8 + ((i + 2) % 5)
            
            # Generate first hexagonal lattice
            lattice1 = hexlattice_gen(r_k=scale1, theta=angle1, order=4, size=img_size)
            
            # Generate second hexagonal lattice
            lattice2 = hexlattice_gen(r_k=scale2, theta=angle2, order=4, size=img_size)
            
            # Convert dask arrays to numpy arrays
            lattice1 = np.array(lattice1).reshape(img_size, img_size)
            lattice2 = np.array(lattice2).reshape(img_size, img_size)
            
            # Create moiré pattern by combining the two lattices
            moire_pattern = np.abs(lattice1 * lattice2)
            
            # Normalize to 0-1 range
            moire_pattern = (moire_pattern - moire_pattern.min()) / (moire_pattern.max() - moire_pattern.min() + 1e-8)
            
            # Apply thresholding to create binary mask with approximately mask_prob ones
            threshold = np.quantile(moire_pattern, 1 - mask_prob)
            binary_mask = (moire_pattern > threshold).astype(np.float32)
            
            # Convert to PyTorch tensor and store
            masks[i] = torch.tensor(binary_mask, device=device)
    
    elif mask_type == "gaussian":
        masks = torch.randn((num_masks, img_size, img_size), device=device)

        # Normalize values to [0,1] range by first computing min/max per mask
        for i in range(num_masks):
            # Min-max normalization for each mask individually
            min_val = masks[i].min()
            max_val = masks[i].max()
            masks[i] = (masks[i] - min_val) / (max_val - min_val + 1e-8)

    elif mask_type == "superres":
        masks = torch.ones((num_masks, img_size, img_size), device=device)
    
    else:
        raise ValueError(f"Unknown mask type: {mask_type}. Choose 'random_binary', 'random_gaussian', 'checkerboard', 'moire', or 'gaussian'.")
    
    return masks  # Shape: [num_masks, img_size, img_size]

# Implement block-wise convolution downsampling using efficient pooling
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
    # avg_pool2d calculates sum / N. divisor_override=1 makes it calculate sum / 1.
    # This directly computes the sum within each block.
    if image.numel() == 0:  # Handle empty tensor case if masking results in empty blocks
         output_h = image.shape[2] // block_size
         output_w = image.shape[3] // block_size
         return torch.zeros(image.shape[0], image.shape[1], output_h, output_w,
                           dtype=image.dtype, device=image.device)

    return F.avg_pool2d(image, kernel_size=block_size, stride=block_size, divisor_override=1)

# Apply the measurement operator A to an image
def measurement_operator(image, masks):
    """
    Apply the measurement operator efficiently using broadcasting and pooling.

    Args:
        image: Tensor of shape [batch_size, channels, img_size, img_size] in [-1, 1] range
        masks: Tensor of shape [num_masks, img_size, img_size]
    
    Returns:
        List of measurements, each of shape [batch_size, channels, output_size, output_size]
    """
    batch_size, channels, img_size, _ = image.shape
    num_masks = masks.shape[0]
    block_size = config["block_size"]

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
    # This maintains compatibility with the rest of the code
    measurements_list = [measurements_batch[:, i] for i in range(num_masks)]
    
    return measurements_list

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

def bicubic_interpolation(target_measurements_batch, batch_size, result_dir=None):
    """
    Implements bicubic interpolation for upsampling measurements.
    target_measurements_batch: List of target measurements from original images, each of shape [batch_size, channels, output_size, output_size]
    """
    print(f"Starting Diffusion Posterior Sampling (DPS) for batch of {batch_size} images")

    # Create output directory for intermediate steps
    if result_dir is None:
        result_dir = config["result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    block_size = config["block_size"]

    target_measurements_batch = torch.stack(target_measurements_batch)  
    target_measurements_batch = target_measurements_batch.permute(1, 0, 2, 3, 4)
    upsampled_images = []
    for i in range(target_measurements_batch.shape[0]):
        zoomed_images = []
        for j in range(target_measurements_batch.shape[1]):
            img_np = target_measurements_batch[i, j, 0].cpu().numpy()
            zoomed_img = zoom(img_np, zoom=block_size, order=3)
            zoomed_images.append(torch.from_numpy(zoomed_img))
        
        # Average over all upsampled images
        zoomed_stack = torch.stack(zoomed_images)
        avg_image = zoomed_stack.mean(dim=0).unsqueeze(0)
        upsampled_images.append(avg_image.to(device))
        
    upsampled_images = torch.stack(upsampled_images)

    return upsampled_images

import prox_tv

def tv_minimize_batch(
    target_measurements_batch,  # list of [batch_size, 1, out, out] tensors
    masks,                      # [num_masks, img_size, img_size]
    img_size=128,
    lmbda=1e-3,
    max_iter=200,
    lr=1e-1
):
    
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    channels = 1

    # Reconstruct all images
    upsampled_images = torch.zeros((batch_size, channels, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)

        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(max_iter):
            optimizer.zero_grad()

            # Forward pass through measurement operator
            simulated_measurements = measurement_operator(x, masks)
            
            # Compute loss 
            simulated_tensor = torch.stack([m[0] for m in simulated_measurements], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(simulated_tensor, target_tensor)

            # Backprop + update
            loss.backward()
            optimizer.step()

            # Proximal step (TV denoising)
            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)
                x_np_tv = prox_tv.tv1_2d(x_np, lmbda)
                x.data = torch.tensor(x_np_tv, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images

import scipy 

def dct2d(x):
    return scipy.fftpack.dct(scipy.fftpack.dct(x.T, norm='ortho').T, norm='ortho')

def idct2d(x):
    return scipy.fftpack.idct(scipy.fftpack.idct(x.T, norm='ortho').T, norm='ortho')

def soft_threshold(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def dct_sparse_minimize_batch(
    target_measurements_batch,  # list of [batch_size, 1, out, out] tensors
    masks,                      # [num_masks, img_size, img_size]
    img_size=128,
    lmbda=1e-3,                 # sparsity threshold
    max_iter=200,
    lr=1e-1
):
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    channels = 1

    upsampled_images = torch.zeros((batch_size, channels, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)

        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(max_iter):
            optimizer.zero_grad()

            # Forward pass
            simulated_measurements = measurement_operator(x, masks)

            # Compute loss
            simulated_tensor = torch.stack([m[0] for m in simulated_measurements], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(simulated_tensor, target_tensor)

            # Backprop
            loss.backward()
            optimizer.step()

            # Proximal step: DCT soft-thresholding
            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)
                dct_coeff = dct2d(x_np)
                dct_thresh = soft_threshold(dct_coeff, lmbda)
                x_np_denoised = idct2d(dct_thresh)
                x.data = torch.tensor(x_np_denoised, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images

import pywt

def wavelet_decompose(x, wavelet='db1', level=None):
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)
    coeffs_flat, coeff_slices = pywt.coeffs_to_array(coeffs)
    return coeffs_flat, coeff_slices, coeffs[0].shape

def wavelet_reconstruct(coeffs_flat, coeff_slices, wavelet='db1'):
    coeffs = pywt.array_to_coeffs(coeffs_flat, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, wavelet=wavelet)

def wavelet_sparse_minimize_batch(
    target_measurements_batch,  # list of [batch_size, 1, out, out] tensors
    masks,                      # [num_masks, img_size, img_size]
    img_size=128,
    lmbda=1e-3,                 # sparsity threshold
    max_iter=200,
    lr=1e-1,
    wavelet='db1',
    level=None
):
    batch_size = target_measurements_batch[0].shape[0]
    num_masks = len(target_measurements_batch)
    device = masks.device
    channels = 1

    upsampled_images = torch.zeros((batch_size, channels, img_size, img_size), device=device)

    for b in range(batch_size):
        x = torch.randn((1, 1, img_size, img_size), requires_grad=True, device=device, dtype=torch.float64)
        optimizer = torch.optim.Adam([x], lr=lr)

        for i in range(max_iter):
            optimizer.zero_grad()

            # Forward pass
            simulated_measurements = measurement_operator(x, masks)

            # Compute loss
            simulated_tensor = torch.stack([m[0] for m in simulated_measurements], dim=0).float()
            target_tensor = torch.stack([target_measurements_batch[m][b] for m in range(num_masks)], dim=0).float()
            loss = F.mse_loss(simulated_tensor, target_tensor)

            # Backprop
            loss.backward()
            optimizer.step()

            # Proximal step: wavelet soft-thresholding
            with torch.no_grad():
                x_np = x.detach().cpu().numpy()[0, 0].astype(np.float64)

                coeffs_flat, coeff_slices, _ = wavelet_decompose(x_np, wavelet, level)
                coeffs_thresh = soft_threshold(coeffs_flat, lmbda)
                x_np_denoised = wavelet_reconstruct(coeffs_thresh, coeff_slices, wavelet)

                # Clip or normalize if needed (optional)
                x.data = torch.tensor(x_np_denoised, device=device).unsqueeze(0).unsqueeze(0)

        upsampled_images[b] = x.detach()

    return upsampled_images


def dmplug_batch(
    model,
    target_measurements_batch,  # list of [batch_size, 1, out, out] tensors
    masks,                      # [num_masks, img_size, img_size]
    img_size=128,                
    max_iter=1000,
    lr=1e-2,
):
    device = masks.device
    channels = config["in_channels"]
    eta = 0
    diff_steps = 3
    dtype = torch.float32
    criterion = torch.nn.MSELoss().to(device)
    stacked = torch.stack(target_measurements_batch, dim=0)
    target_measurements_batch = stacked.permute(1, 0, 2, 3, 4)

    # Define the DDIM scheduler
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(diff_steps)
    num_images = target_measurements_batch.shape[0]
    y_n = target_measurements_batch.view(-1, channels, target_measurements_batch.shape[3], target_measurements_batch.shape[4])
    # print('y', y_n.shape)
    y_n.requires_grad = False

    # upsampled_images = torch.zeros((batch_size, channels, img_size, img_size), device=device)

    # Initialize from Gaussian noise
    Z = torch.randn((num_images, channels, img_size, img_size), device=device, dtype=dtype, requires_grad=True)
    criterion = torch.nn.MSELoss().to(device)
    params_group1 = {'params': Z, 'lr': lr}
    optimizer = torch.optim.Adam([params_group1])

    # Run DMPlug
    pbar = tqdm(range(max_iter))
    for t in pbar:
        model.eval()
        optimizer.zero_grad()

        for i, tt in enumerate(scheduler.timesteps):
            t_i = (torch.ones(1) * tt).cuda()
            if i == 0:
                noise_pred = model(Z, t_i).sample
            else:
                noise_pred = model(x_t, t_i).sample

            if i == 0:
                x_t = scheduler.step(noise_pred, tt, Z, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
            else:
                x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample

        output = torch.clamp(x_t, -1, 1)
        # print('output', output.shape)
        # print('x_t', x_t.shape)
        pred_measurements = measurement_operator(output, masks)
        stacked = torch.stack(pred_measurements, dim=0)
        pred_measurements = stacked.permute(1, 0, 2, 3, 4)
        pred_measurements = pred_measurements.view(-1, config["in_channels"], pred_measurements.shape[3], pred_measurements.shape[4])
        loss = criterion(pred_measurements, y_n)
        loss.backward()
        optimizer.step()

        # Update progress info
        pbar.set_description(f"Step: {t}, MSE: {loss.item():.6f}")

    return x_t.detach()



# # Diffusion Posterior Sampling algorithm (DPS)
# def diffusion_posterior_sampling(target_measurements_batch, masks, batch_size, current_batch_ids, unet, scheduler, num_timesteps=1000, zeta_scale=1e-1, beta=0.9, result_dir=None):
#     """
#     Implement the Diffusion Posterior Sampling algorithm using DDPM sampling with momentum.
    
#     Args:
#         target_measurements_batch: List of target measurements from original images, each of shape [batch_size, channels, output_size, output_size]
#         masks: Binary masks used for measurements
#         batch_size: Number of images in the batch
#         current_batch_ids: List of image IDs in the current batch for saving intermediate results
#         unet: UNet model from the pipeline
#         scheduler: DDPM scheduler from the pipeline
#         num_timesteps: Number of diffusion timesteps
#         zeta_scale: Scale factor for the step size
#         beta: Momentum factor (default: 0.9)
#         result_dir: Directory to save results
    
#     Returns:
#         Reconstructed images in [-1, 1] range with shape [batch_size, channels, img_size, img_size]
#     """
#     print(f"Starting Diffusion Posterior Sampling (DPS) for batch of {batch_size} images with momentum (beta={beta})...")
    
#     # Create output directory for intermediate steps
#     if result_dir is None:
#         result_dir = config["result_dir"]
#     os.makedirs(result_dir, exist_ok=True)
    
#     # Initialize from Gaussian noise (already in the correct range for the diffusion model)
#     x_T = torch.randn(batch_size, config["in_channels"], config["img_size"], config["img_size"]).to(device)
#     x_t = x_T.clone()
    
#     # Initialize momentum buffer
#     momentum = torch.zeros_like(x_T)
    
#     # Initializing DDPM Sampler for DPS
#     print("Initializing DDPM Sampler for DPS...")
#     # Set the number of inference steps
#     scheduler.set_timesteps(num_timesteps)
    
#     # Get DDPM timesteps (usually high to low)
#     timesteps = scheduler.timesteps
#     pbar = tqdm(timesteps)
    
#     # Iterate using DDPM timesteps
#     for t in pbar:
#         # Calculate step_idx (0 to num_timesteps-1) for zeta schedule compatibility
#         # pbar.n goes from 0 to N-1. We need N-1 down to 0.
#         step_idx = (pbar.total - 1) - pbar.n
        
#         # Ensure t is a tensor for model input if needed by the unet
#         timestep_tensor = t if isinstance(t, torch.Tensor) else torch.tensor([t], device=device)
        
#         # 1. Compute noise prediction
#         with torch.no_grad():
#             noise_pred = unet(x_t, timestep_tensor).sample
        
#         # 2. Use the DDPM scheduler step function
#         # It computes the previous sample (x_{t-1}) and predicted original sample (x0)
#         step_output = scheduler.step(noise_pred, t, x_t)
#         x_prev_ddpm = step_output.prev_sample  # DDPM estimate for x_{t-1}
#         x0_estimate_ddpm = step_output.pred_original_sample  # DDPM estimate for x0
        
#         # 3. Clamp the estimated x0
#         x0_estimate_ddpm = torch.clamp(x0_estimate_ddpm, -1, 1)
        
#         # 4. Compute the measurement consistency gradient using DDPM's x0 estimate
#         grad_batch = measurement_consistency_gradient(x0_estimate_ddpm, target_measurements_batch, masks)
        
#         # Calculate residual norm using DDPM's x0 estimate
#         with torch.no_grad():
#             pred_measurements = measurement_operator(x0_estimate_ddpm, masks)
            
#             # Stack measurements and compute MSE
#             pred_stacked = torch.stack(pred_measurements)
#             target_stacked = torch.stack(target_measurements_batch)
            
#             # Calculate MSE per image in batch
#             batch_mse = F.mse_loss(pred_stacked, target_stacked, reduction='none')
#             mse_per_image = batch_mse.mean(dim=[0, 2, 3, 4]).cpu().numpy()  # Mean over masks, channels, spatial dims
#             residual_norm = np.sqrt(mse_per_image)  # L2 norm per image
#             mean_residual = np.mean(residual_norm)  # Average for progress reporting
        
#         # --- Nesterov Accelerated Gradient --- 
#         # 1. Calculate lookahead x0 estimate
#         x0_lookahead = x0_estimate_ddpm - beta * momentum

#         # 2. Compute gradient at the lookahead point
#         grad_batch_nesterov = measurement_consistency_gradient(x0_lookahead, target_measurements_batch, masks)

#         # 3. Update momentum using the Nesterov gradient
#         momentum = beta * momentum + zeta_scale * grad_batch_nesterov

#         # 4. Apply gradient guidance using the updated momentum
#         x_prev_guided = x_prev_ddpm - momentum
#         # --- End Nesterov --- 

#         # Update x_t for the next iteration
#         x_t = x_prev_guided
        
#         # Update progress info
#         t_value = t.item() if isinstance(t, torch.Tensor) else t
#         # Report the Nesterov gradient norm in the progress bar for clarity
#         pbar.set_description(f"Step: {step_idx} (t={t_value}), Zeta: {zeta_scale:.2f}, Mean MSE: {mean_residual:.6f}, Mean Grad (NAG): {torch.mean(grad_batch_nesterov).item():.6f}")
        
#         # Clean up memory every 100 steps
#         if step_idx % 100 == 0:
#             torch.cuda.empty_cache()
    
#     return x_t

# Function to save metrics to CSV
def save_metrics_to_csv(metrics, result_dir, experiment_params=None):
    """
    Save reconstruction metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics (MSE, MAE, PSNR, SSIM, LPIPS, L2_norm, Avg_measurement_MSE)
        result_dir: Directory to save the CSV file
        experiment_params: Dictionary of experiment parameters (optional)
    """
    csv_path = os.path.join(result_dir, "reconstruction_metrics.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['image_id', 'block_size', 'num_masks', 'mask_type', 'MSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'L2_norm', 'Avg_measurement_MSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = metrics.copy()
        if experiment_params:
            row.update(experiment_params)
        
        writer.writerow(row)
    
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

# Analyze reconstruction quality
def analyze_reconstruction(original_image_batch, reconstructed_image_batch, target_measurements_batch, masks, current_batch_ids, result_dir=None, experiment_params=None):
    """
    Analyze the quality of the reconstruction for a batch of images.
    
    Args:
        original_image_batch: Original image tensor [batch_size, channels, img_size, img_size] in [-1, 1] range
        reconstructed_image_batch: Reconstructed image tensor [batch_size, channels, img_size, img_size] in [-1, 1] range
        target_measurements_batch: List of target measurements for the batch
        masks: Binary masks used for measurements
        current_batch_ids: List of image IDs in the current batch
        result_dir: Directory to save results
        experiment_params: Dictionary of experiment parameters
        
    Returns:
        List of dictionaries containing all computed metrics for each image in the batch
    """
    if result_dir is None:
        result_dir = config["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize LPIPS loss function
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    batch_size = original_image_batch.shape[0]
    all_metrics = []
    
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
        reconstructed_measurements = measurement_operator(reconstructed_image, masks)
        
        measurement_mses = []
        for i in range(len(target_measurements_single)):
            mse_i = F.mse_loss(reconstructed_measurements[i], target_measurements_single[i]).item()
            measurement_mses.append(mse_i)
        
        avg_measurement_mse = sum(measurement_mses) / len(measurement_mses)
        print(f"Average Measurement MSE: {avg_measurement_mse:.6f}")
        
        # 3. Visualize results
        plt.figure(figsize=(12, 6))
        
        # For multi-channel images, we'll visualize them differently
        if original_image.shape[1] > 1:  # Multi-channel
            # Original image
            plt.subplot(1, 3, 1)
            # Take first 3 channels for RGB or just first channel for grayscale
            img = original_vis[0, :min(3, original_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
            if original_vis.shape[1] == 1:
                plt.imshow(original_vis[0, 0].cpu().numpy(), cmap='gray')
            else:
                plt.imshow(np.clip(img, 0, 1))
            plt.title("Original Image")
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(1, 3, 2)
            img = reconstructed_vis[0, :min(3, reconstructed_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
            if reconstructed_vis.shape[1] == 1:
                plt.imshow(reconstructed_vis[0, 0].cpu().numpy(), cmap='gray')
            else:
                plt.imshow(np.clip(img, 0, 1))
            plt.title("Reconstructed Image")
            plt.axis('off')
            
            # Error map - use mean across channels for multi-channel
            plt.subplot(1, 3, 3)
            error_map = torch.abs(original_image - reconstructed_image).mean(dim=1).squeeze().cpu().numpy()
            plt.imshow(error_map, cmap='hot')
            plt.colorbar(label='Absolute Error')
            plt.title("Error Map")
            plt.axis('off')
        else:
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_vis.squeeze().cpu().numpy(), cmap='gray')
            plt.title("Original Image")
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(1, 3, 2)
            plt.imshow(reconstructed_vis.squeeze().cpu().numpy(), cmap='gray')
            plt.title("Reconstructed Image")
            plt.axis('off')
            
            # Absolute error map
            plt.subplot(1, 3, 3)
            error_map = torch.abs(original_image - reconstructed_image).squeeze().cpu().numpy()
            plt.imshow(error_map, cmap='hot')
            plt.colorbar(label='Absolute Error')
            plt.title("Error Map")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{result_dir}/reconstruction_comparison_image_{image_id}.png")
        plt.close()
        
        # 4. Save metrics to CSV
        metrics = {
            'image_id': image_id,
            'MSE': mse,
            'MAE': mae,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips_value,
            'L2_norm': l2_norm,
            'Avg_measurement_MSE': avg_measurement_mse
        }
        all_metrics.append(metrics)
        save_metrics_to_csv(metrics, result_dir, experiment_params)
    
    return all_metrics

def main():
    # Allow command-line arguments to override default configuration
    import argparse
    parser = argparse.ArgumentParser(description='Run Diffusion Posterior Sampling with dynamic parameters')
    parser.add_argument('--model', type=str, default='google/ddpm-cifar10-32', 
                        help='Model name/path (default: google/ddpm-cifar10-32)')
    parser.add_argument('--block_size', type=int, default=config["block_size"], 
                        help=f'Block size for downsampling (default: {config["block_size"]})')
    parser.add_argument('--num_masks', type=int, default=64, 
                        help='Number of binary masks to use (default: 30)')
    parser.add_argument('--mask_prob', type=float, default=0.5, 
                        help='Probability for binary mask generation (default: 0.5)')
    parser.add_argument('--mask_type', type=str, default="random_binary", choices=["random_binary", "random_gaussian", "checkerboard", "moire", "gaussian", "superres"],
                        help='Type of mask to use - "random_binary", "random_gaussian", "checkerboard", "moire", or "gaussian" (default: "random_binary")')
    parser.add_argument('--cryoem_path', type=str, 
                        default="/usr/scratch/CryoEM/CryoSensing/ribosomes/inputs_new/val.pt",
                        help='Path to the CryoEM dataset file (.pt format)')
    parser.add_argument('--start_id', type=int, default=0,
                        help='Starting index of the image range to process (default: 0)')
    parser.add_argument('--end_id', type=int, default = 1,
                        help='Ending index of the image range to process (inclusive)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of images to process in each batch (default: 4)')
    parser.add_argument('--use_cryoem', action='store_true',
                        help='Use CryoEM data instead of generating image with DDPM')
    parser.add_argument('--result_dir', type=str, default="results",
                        help='Directory to save results (default: results)')
    parser.add_argument('--baseline', type=str, default="bicubic")
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=1e-1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--noise_level', type=float, default=0.0,
        help='Standard deviation of Gaussian noise to add to measurements (default: 0.0)')
    
    args = parser.parse_args()
    
    # Ensure required packages are installed
    try:
        import lpips
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "lpips", "scikit-image"])
        import lpips
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        print("Required packages installed successfully.")
    
    # Update configuration with command-line arguments
    config["block_size"] = args.block_size
    config["result_dir"] = args.result_dir
    
    # Create results directory
    os.makedirs(config["result_dir"], exist_ok=True)
    print(f"Results will be saved to: {config['result_dir']}")

    # Initialize DDPM pipeline after parsing arguments
    print("Initializing DDPM pipeline with model")
    pipeline = DDPMPipeline.from_pretrained(args.model).to(device)
    model = pipeline.unet
    model_config = model.config
    in_channels = model_config.in_channels
    if hasattr(model_config, "sample_size"):
        img_size = model_config.sample_size
    
    # Update configuration with determined dimensions
    in_channels = args.in_channels
    config.update({
        "img_size": img_size,
        "in_channels": in_channels,
        "output_size": img_size // config["block_size"]  # Size after block-wise downsampling
    })
    
    print(f"Image size: {config['img_size']}x{config['img_size']}")
    print(f"Block size: {config['block_size']}x{config['block_size']}")
    print(f"Output size after downsampling: {config['output_size']}x{config['output_size']}")
    print(f"Using mask type: {args.mask_type}")
    print(f"Noise level for measurements (sigma): {args.noise_level}")
    
    # Ensure end_id is provided if using CryoEM data
    if args.use_cryoem:
        if args.end_id is None:
            raise ValueError("--end_id must be provided when using --use_cryoem")
        if args.end_id < args.start_id:
            raise ValueError("--end_id must be greater than or equal to --start_id")
    
    # Create binary masks (do this once and use for all batches)
    print(f"Creating {args.num_masks} binary masks...")
    masks = create_binary_masks(
        num_masks=args.num_masks, 
        mask_prob=args.mask_prob, 
        mask_type=args.mask_type, 
        result_dir=config["result_dir"]
    )
    
    # Process images in batches
    if args.use_cryoem:
        print(f"Processing CryoEM images from index {args.start_id} to {args.end_id}")
        total_images = args.end_id - args.start_id + 1
        print(f"Total number of images to process: {total_images}")
        print(f"Processing in batches of size: {args.batch_size}")
        
        # Calculate number of batches
        num_batches = math.ceil(total_images / args.batch_size)
        
        # Create CSV file header for metrics
        csv_path = os.path.join(config["result_dir"], "reconstruction_metrics.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['image_id', 'block_size', 'num_masks', 'mask_type', 'MSE', 'MAE', 'PSNR', 'SSIM', 'LPIPS', 'L2_norm', 'Avg_measurement_MSE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        # Initialize LPIPS loss function outside the loop to save time
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        
        # Iterate through image range in batches
        for batch_idx in range(num_batches):
            batch_start = args.start_id + batch_idx * args.batch_size
            batch_end = min(args.start_id + (batch_idx + 1) * args.batch_size - 1, args.end_id)
            current_batch_size = batch_end - batch_start + 1
            current_batch_ids = list(range(batch_start, batch_end + 1))
            
            print(f"\n===== Processing Batch {batch_idx + 1}/{num_batches} =====")
            print(f"Batch image IDs: {current_batch_ids}")
            
            # 1. Load batch of CryoEM images
            print(f"Loading batch of CryoEM images...")
            print('DIRECTORY', args.cryoem_path)
            original_image_batch = load_cryoem_batch(
                args.cryoem_path, 
                current_batch_ids, 
                config["img_size"], 
                device, 
                result_dir=config["result_dir"]
            )
            print('DIRECTORY', original_image_batch[0].shape)
            
            # 2. Get measurements from the batch of original images
            print("Getting measurements from batch of original images...")
            target_measurements_batch = measurement_operator(original_image_batch, masks)

            # Apply noise to measurements if specified
            if args.noise_level > 0:
                print(f"Adding Gaussian noise with sigma={args.noise_level} to measurements...")
                target_measurements_batch = add_gaussian_noise(target_measurements_batch, args.noise_level)

            # 3. Perform baseline for reconstruction
            if args.baseline == "bicubic":
                print("Performing Bicubic Interpolation...")
                reconstructed_image_batch = bicubic_interpolation(
                    target_measurements_batch=target_measurements_batch,
                    batch_size=current_batch_size,
                    result_dir=config["result_dir"]
                )
            elif args.baseline == "tv_minimize":
                print("Performing TV Minimization...")
                reconstructed_image_batch = tv_minimize_batch(
                    target_measurements_batch=target_measurements_batch,
                    masks=masks,
                    lr=args.learning_rate,
                    lmbda=args.lambda_,
                    max_iter=args.num_epochs
                )
            elif args.baseline == 'dct':
                print("Performing DCT Minimization...")
                reconstructed_image_batch = dct_sparse_minimize_batch(
                    target_measurements_batch=target_measurements_batch,
                    masks=masks,
                    lr=args.learning_rate,
                    lmbda=args.lambda_,
                    max_iter=args.num_epochs
                )
            elif args.baseline == 'wavelet':
                print("Performing Wavelet Minimization...")
                reconstructed_image_batch = wavelet_sparse_minimize_batch(
                    target_measurements_batch=target_measurements_batch,
                    masks=masks,
                    lr=args.learning_rate,
                    lmbda=args.lambda_,
                    max_iter=args.num_epochs
                )
            elif args.baseline == 'dmplug':
                print("Performing DMPlug...")
                max_iter = 1000

                reconstructed_image_batch = dmplug_batch(
                    model=model,
                    target_measurements_batch=target_measurements_batch,
                    masks=masks,
                    img_size=config["img_size"],
                    max_iter=max_iter
                )

            # print(len(reconstructed_image_batch), reconstructed_image_batch[0].shape)
            
            # 4. Process each image in the batch
            for j, image_id in enumerate(current_batch_ids):
                print(f"\n----- Processing results for image {image_id} -----")
                
                # Extract single images from batch
                original_image = original_image_batch[j:j+1]  # Keep batch dimension for LPIPS
                reconstructed_image = reconstructed_image_batch[j:j+1]
                
                # Extract single measurements for this image from the batch
                target_measurements_single = [m_batch[j:j+1] for m_batch in target_measurements_batch]
                
                # Save the raw reconstructed tensor (in [-1,1] range) as .pt file
                torch.save(reconstructed_image.cpu(), f"{config['result_dir']}/reconstruction_raw_image_{image_id}.pt")
                
                # 5. Calculate metrics
                # Pixel-wise metrics
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
                
                # Calculate measurement-wise error
                reconstructed_measurements = measurement_operator(reconstructed_image, masks)
                
                measurement_mses = []
                for i in range(len(target_measurements_single)):
                    mse_i = F.mse_loss(reconstructed_measurements[i], target_measurements_single[i]).item()
                    measurement_mses.append(mse_i)
                
                avg_measurement_mse = sum(measurement_mses) / len(measurement_mses)
                
                # Print key metrics
                print(f"PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, MSE: {mse:.6f}")
                
                # 6. Create and save comparison visualization
                plt.figure(figsize=(12, 6))
                
                # For multi-channel images, we'll visualize them differently
                if original_image.shape[1] > 1:  # Multi-channel
                    # Original image
                    plt.subplot(1, 3, 1)
                    # Take first 3 channels for RGB or just first channel for grayscale
                    img = original_vis[0, :min(3, original_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
                    if original_vis.shape[1] == 1:
                        plt.imshow(original_vis[0, 0].cpu().numpy(), cmap='gray')
                    else:
                        plt.imshow(np.clip(img, 0, 1))
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    # Reconstructed image
                    plt.subplot(1, 3, 2)
                    img = reconstructed_vis[0, :min(3, reconstructed_vis.shape[1])].permute(1, 2, 0).cpu().numpy()
                    if reconstructed_vis.shape[1] == 1:
                        plt.imshow(reconstructed_vis[0, 0].cpu().numpy(), cmap='gray')
                    else:
                        plt.imshow(np.clip(img, 0, 1))
                    plt.title("Reconstructed Image")
                    plt.axis('off')
                    
                    # Error map - use mean across channels for multi-channel
                    plt.subplot(1, 3, 3)
                    error_map = torch.abs(original_image - reconstructed_image).mean(dim=1).squeeze().cpu().numpy()
                    plt.imshow(error_map, cmap='hot')
                    plt.colorbar(label='Absolute Error')
                    plt.title("Error Map")
                    plt.axis('off')
                else:
                    # Original image
                    plt.subplot(1, 3, 1)
                    plt.imshow(original_vis.squeeze().cpu().numpy(), cmap='gray')
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    # Reconstructed image
                    plt.subplot(1, 3, 2)
                    plt.imshow(reconstructed_vis.squeeze().cpu().numpy(), cmap='gray')
                    plt.title("Reconstructed Image")
                    plt.axis('off')
                    
                    # Error map
                    plt.subplot(1, 3, 3)
                    error_map = torch.abs(original_image - reconstructed_image).squeeze().cpu().numpy()
                    plt.imshow(error_map, cmap='hot')
                    plt.colorbar(label='Absolute Error')
                    plt.title("Error Map")
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{config['result_dir']}/comparison_image_{image_id}.png")
                plt.close()
                
                # 7. Save metrics to CSV
                metrics = {
                    'image_id': image_id,
                    'block_size': args.block_size,
                    'num_masks': args.num_masks,
                    'mask_type': args.mask_type,
                    'MSE': mse,
                    'MAE': mae,
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'LPIPS': lpips_value,
                    'L2_norm': l2_norm,
                    'Avg_measurement_MSE': avg_measurement_mse
                }
                
                # Append metrics to CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(metrics)
            
            # 8. Clean up memory between batches
            del original_image_batch, target_measurements_batch, reconstructed_image_batch
            if 'grad_batch' in locals(): 
                del grad_batch
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cleaned up memory after batch {batch_idx + 1}")
            
        print(f"\n===== Completed processing all {total_images} images =====")
        print(f"Results saved in the '{config['result_dir']}' directory.")
        


"""
Diffusion functions
"""
# This code is from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, deprecate
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        next_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t+1}) of previous timestep. `next_sample` should be used as next model input in the
            reverse denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: Optional[torch.FloatTensor] = None
    next_sample: Optional[torch.FloatTensor] = None
    pred_original_sample: Optional[torch.FloatTensor] = None


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`~ConfigMixin`] also provides general loading and saving functionality via the [`~ConfigMixin.save_config`] and
    [`~ConfigMixin.from_config`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between -1 and 1 for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.

    """

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[np.ndarray] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = torch.from_numpy(trained_betas)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.
        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset
        
    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): TODO
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the model_output is always re-derived from the clipped x_0 in Glide
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            device = model_output.device if torch.is_tensor(model_output) else "cpu"
            noise = torch.randn(model_output.shape, generator=generator).to(device)
            variance = self._get_variance(timestep, prev_timestep) ** (0.5) * eta * noise

            prev_sample = prev_sample + variance

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def reverse_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): TODO
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        next_timestep = min(self.config.num_train_timesteps - 2,
                            timestep + self.config.num_train_timesteps // self.num_inference_steps)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_next = self.alphas_cumprod[next_timestep] if next_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 4. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 5. TODO: simple noising implementatiom
        next_sample = self.add_noise(pred_original_sample,
                                     model_output,
                                     torch.LongTensor([next_timestep]))

        # # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        # variance = self._get_variance(next_timestep, timestep)
        # std_dev_t = eta * variance ** (0.5)

        # if use_clipped_model_output:
        #     # the model_output is always re-derived from the clipped x_0 in Glide
        #     model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # pred_sample_direction = (1 - alpha_prod_t_next - std_dev_t**2) ** (0.5) * model_output

        # # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # next_sample = alpha_prod_t_next ** (0.5) * pred_original_sample + pred_sample_direction

        if not return_dict:
            return (next_sample,)

        return DDIMSchedulerOutput(next_sample=next_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        if self.alphas_cumprod.device != original_samples.device:
            self.alphas_cumprod = self.alphas_cumprod.to(original_samples.device)
        if timesteps.device != original_samples.device:
            timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps



 



 

if __name__ == "__main__":
    main()