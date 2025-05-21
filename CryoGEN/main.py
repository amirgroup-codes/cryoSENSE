"""
Main API for using CryoGEN algorithm.
"""

import torch
import os
import numpy as np
from diffusers import DDPMPipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from PIL import Image

from .core import measurement_operator, cryogen_sampling
from .masks import create_binary_masks
from .data import load_cryoem_image, load_cryoem_batch, add_gaussian_noise
from .evaluation import plot_measurements_and_original, analyze_reconstruction
from .config import get_recommended_params, load_config

class CryoGEN:
    """
    Main class for using the CryoGEN algorithm for CryoEM image reconstruction.
    """
    
    def __init__(self,
                 model_path=None,
                 block_size=4,
                 img_size=None,
                 in_channels=None,
                 device="cuda",
                 result_dir="results",
                 verbose=False,
                 use_config=True):
        """
        Initialize the CryoGEN model.
        
        Args:
            model_path: Path to pretrained DDPM model
            block_size: Block size for downsampling
            img_size: Image size (derived from model if not provided)
            in_channels: Number of channels (derived from model if not provided)
            device: Computation device (cuda or cpu)
            result_dir: Directory to save results
            verbose: Enable verbose output and visualization
            use_config: Use recommended configuration from file based on block_size
        """
        self.block_size = block_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.result_dir = result_dir
        self.verbose = verbose
        self.use_config = use_config
        
        # Load recommended parameters if requested
        if self.use_config:
            config = load_config(block_size=self.block_size)
            self.zeta_scale = config.get("zeta_scale", 1.0 if block_size <= 16 else 10.0)
            self.zeta_min = config.get("zeta_min", 1e-2)
            self.beta = config.get("beta", 0.9)
            self.beta_min = config.get("beta_min", 0.1)
            print(f"Loaded recommended parameters from configuration: zeta_scale={self.zeta_scale}, zeta_min={self.zeta_min}, beta={self.beta}, beta_min={self.beta_min}")
        
        # Create results directory
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Create subdirectories for verbose mode
        if self.verbose:
            os.makedirs(os.path.join(self.result_dir, "masks"), exist_ok=True)
            os.makedirs(os.path.join(self.result_dir, "measurements"), exist_ok=True)
            os.makedirs(os.path.join(self.result_dir, "diffusion_process"), exist_ok=True)
        
        # Initialize model
        if model_path:
            print(f"Loading DDPM model from: {model_path}")
            self.pipeline = DDPMPipeline.from_pretrained(model_path).to(self.device)
            self.scheduler = self.pipeline.scheduler
            self.unet = self.pipeline.unet
            
            # Get image dimensions from model
            model_config = self.unet.config
            self.in_channels = model_config.in_channels if in_channels is None else in_channels
            
            # For most diffusion models, the image dimensions can be determined from the sample_size
            if hasattr(model_config, "sample_size"):
                self.img_size = model_config.sample_size if img_size is None else img_size
            else:
                # Default to a standard size if not specified
                self.img_size = 128 if img_size is None else img_size
        else:
            self.pipeline = None
            self.scheduler = None
            self.unet = None
            self.in_channels = in_channels or 1
            self.img_size = img_size or 128
            
        self.output_size = self.img_size // self.block_size
        
        print(f"CryoGEN initialized with:")
        print(f"  - Image size: {self.img_size}x{self.img_size}")
        print(f"  - Input channels: {self.in_channels}")
        print(f"  - Block size: {self.block_size}x{self.block_size}")
        print(f"  - Output size after downsampling: {self.output_size}x{self.output_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Verbose mode: {'Enabled' if self.verbose else 'Disabled'}")
        print(f"  - Using config: {'Yes' if self.use_config else 'No'}")
        print(f"  - Results directory: {self.result_dir}")
    
    def load_model(self, model_path):
        """
        Load a DDPM model.
        
        Args:
            model_path: Path to pretrained DDPM model
        """
        print(f"Loading DDPM model from: {model_path}")
        self.pipeline = DDPMPipeline.from_pretrained(model_path).to(self.device)
        self.scheduler = self.pipeline.scheduler
        self.unet = self.pipeline.unet
        
        # Update image dimensions from model
        model_config = self.unet.config
        self.in_channels = model_config.in_channels
        
        if hasattr(model_config, "sample_size"):
            self.img_size = model_config.sample_size
        
        self.output_size = self.img_size // self.block_size
        print(f"Model loaded successfully. Image size: {self.img_size}x{self.img_size}, Channels: {self.in_channels}")
    
    def generate_masks(self, num_masks=30, mask_prob=0.5, mask_type="random_binary"):
        """
        Generate binary masks for measurement.
        
        Args:
            num_masks: Number of masks to generate
            mask_prob: Probability for binary mask generation
            mask_type: Type of mask ("random_binary", "random_gaussian", "checkerboard", "moire")
            
        Returns:
            Generated masks
        """
        print(f"Generating {num_masks} masks of type '{mask_type}'...")
        masks = create_binary_masks(
            num_masks=num_masks,
            mask_prob=mask_prob,
            mask_type=mask_type,
            img_size=self.img_size,
            device=self.device
        )
        
        # Save all masks if verbose mode is enabled
        if self.verbose:
            mask_dir = os.path.join(self.result_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)
            
            # Show only 10 masks
            for i in range(min(10, len(masks))):
                mask = masks[i].cpu().numpy()
                
                plt.figure(figsize=(8, 8), frameon=False)
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                plt.savefig(f"{mask_dir}/mask_{i}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
            
            print(f"All {len(masks)} masks saved to {mask_dir}/")
            
        return masks
    
    def save_measurements(self, measurements, batch_size):
        """
        Save all measurements as images.
        
        Args:
            measurements: List of measurements
            batch_size: Number of images in the batch
        """
        if not self.verbose:
            return
            
        measurement_dir = os.path.join(self.result_dir, "measurements")
        os.makedirs(measurement_dir, exist_ok=True)
        
        # Process each image in the batch
        for b in range(batch_size):
            # Save all measurements for this image (only show first 10)
            for i in range(min(10, len(measurements))):
                # Get the measurement for this mask
                meas = measurements[i][b]
                
                # Convert from arbitrary range to [0, 1] for visualization
                meas_vis = (meas - meas.min()) / (meas.max() - meas.min() + 1e-8)
                
                plt.figure(figsize=(8, 8), frameon=False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                if meas.shape[0] == 1:  # Single channel
                    plt.imshow(meas_vis[0].cpu().numpy(), cmap='gray')
                else:  # RGB (show first channel only for simplicity)
                    plt.imshow(meas_vis[0].cpu().numpy(), cmap='gray')
                
                plt.axis('off')
                plt.savefig(f"{measurement_dir}/img{b}_measurement_{i}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
                
        print(f"All measurements saved to {measurement_dir}/")
    
    def diffusion_process_callback(self, image_ids, save_every=10):
        """
        Creates a callback function for saving diffusion steps.
        
        Args:
            image_ids: List of image IDs
            save_every: Save frequency for timesteps
            
        Returns:
            Callback function
        """
        if not self.verbose:
            return None
            
        # Dictionary to store frames for each image
        frames = {img_id: [] for img_id in image_ids}
        
        def callback(timestep, x_t):
            if timestep % save_every != 0:
                return
                
            # Save the current state for each image in the batch
            for i, img_id in enumerate(image_ids):
                # Convert from [-1, 1] to [0, 1] for visualization
                img = ((x_t[i] + 1) / 2).clamp(0, 1)
                
                # Save to frames dictionary
                img_np = img[0].cpu().numpy() if self.in_channels == 1 else img.permute(1, 2, 0).cpu().numpy()
                frames[img_id].append(img_np)
                
                # Also save individual frame as an image
                plt.figure(figsize=(8, 8), frameon=False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                
                if self.in_channels == 1:
                    plt.imshow(img[0].cpu().numpy(), cmap='gray')
                else:
                    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
                    
                plt.axis('off')
                plt.savefig(f"{self.result_dir}/diffusion_process/img{img_id}_t{timestep}.png", 
                           bbox_inches='tight', pad_inches=0)
                plt.close()
        
        # Store the frames dictionary as an attribute for later access
        self.diffusion_frames = frames
        
        return callback
    
    def create_diffusion_gifs(self, image_ids):
        """
        Create GIFs from saved diffusion steps.
        
        Args:
            image_ids: List of image IDs
        """
        if not self.verbose or not hasattr(self, 'diffusion_frames'):
            return
            
        print("Creating GIFs of the diffusion process...")
        
        # Create a GIF for each image
        for img_id in image_ids:
            if img_id not in self.diffusion_frames or not self.diffusion_frames[img_id]:
                continue
                
            frames = self.diffusion_frames[img_id]
            
            # Convert frames to PIL Images
            pil_frames = []
            for frame in frames:
                if frame.ndim == 2:  # Grayscale
                    pil_frame = Image.fromarray((frame * 255).astype(np.uint8), mode='L')
                else:  # RGB
                    pil_frame = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
                pil_frames.append(pil_frame)
            
            # Save as GIF
            gif_path = f"{self.result_dir}/diffusion_process_img{img_id}.gif"
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                optimize=False,
                duration=100,  # Time between frames in milliseconds
                loop=0  # Loop forever
            )
            
            print(f"GIF saved to {gif_path}")
    
    def get_measurements(self, image, masks):
        """
        Apply the measurement operator to get measurements.
        
        Args:
            image: Input image tensor [batch_size, channels, img_size, img_size]
            masks: Binary masks tensor [num_masks, img_size, img_size]
            
        Returns:
            List of measurements
        """
        return measurement_operator(image, masks, self.block_size)
    
    def reconstruct_image(self, 
                         target_measurements, 
                         masks, 
                         batch_size=1, 
                         num_timesteps=1000, 
                         zeta_scale=None, 
                         zeta_min=None, 
                         beta=None, 
                         beta_min=None,
                         image_ids=None):
        """
        Reconstruct images from measurements using CryoGEN.
        
        Args:
            target_measurements: List of target measurements
            masks: Binary masks used for measurements
            batch_size: Number of images in the batch
            num_timesteps: Number of diffusion timesteps
            zeta_scale: Final scale factor for the step size
            zeta_min: Initial scale factor for the step size
            beta: Final momentum factor
            beta_min: Initial momentum factor
            image_ids: List of image IDs (for verbose mode)
            
        Returns:
            Reconstructed images tensor
        """
        if self.unet is None or self.scheduler is None:
            raise ValueError("DDPM model not loaded. Call load_model() first or initialize with a model_path.")
        
        # Use parameters from config if available and not provided
        if self.use_config:
            zeta_scale = zeta_scale if zeta_scale is not None else self.zeta_scale
            zeta_min = zeta_min if zeta_min is not None else self.zeta_min
            beta = beta if beta is not None else self.beta
            beta_min = beta_min if beta_min is not None else self.beta_min
        else:
            # Default values if not provided and not using config
            zeta_scale = zeta_scale if zeta_scale is not None else (1.0 if self.block_size <= 16 else 10.0)
            zeta_min = zeta_min if zeta_min is not None else 1e-2
            beta = beta if beta is not None else 0.9
            beta_min = beta_min if beta_min is not None else 0.1
            
        print(f"Reconstruction parameters: zeta_scale={zeta_scale}, zeta_min={zeta_min}, beta={beta}, beta_min={beta_min}")
        print("Starting image reconstruction with CryoGEN...")
        
        # Create callback for saving diffusion steps if in verbose mode
        callback = self.diffusion_process_callback(image_ids) if self.verbose else None
        
        reconstructed_images = cryogen_sampling(
            target_measurements,
            masks,
            batch_size,
            self.unet,
            self.scheduler,
            self.block_size,
            num_timesteps=num_timesteps,
            zeta_scale=zeta_scale,
            zeta_min=zeta_min,
            beta=beta,
            beta_min=beta_min,
            device=self.device,
            callback=callback
        )
        
        # Create GIFs from saved frames if in verbose mode
        if self.verbose and image_ids:
            self.create_diffusion_gifs(image_ids)
        
        return reconstructed_images
    
    def evaluate_reconstruction(self, original_images, reconstructed_images, target_measurements, masks, image_ids=None):
        """
        Evaluate the quality of the reconstruction.
        
        Args:
            original_images: Original image tensor
            reconstructed_images: Reconstructed image tensor
            target_measurements: List of target measurements
            masks: Binary masks used for measurements
            image_ids: List of image IDs (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("Evaluating reconstruction quality...")
        metrics = analyze_reconstruction(
            original_images,
            reconstructed_images,
            target_measurements,
            masks,
            self.block_size,
            current_batch_ids=image_ids,
            result_dir=self.result_dir
        )
        
        return metrics
    
    def reconstruct_from_cryoem(self, 
                               file_path, 
                               image_ids=None, 
                               num_masks=30, 
                               mask_prob=0.5, 
                               mask_type="random_binary", 
                               noise_level=0.0,
                               num_timesteps=1000, 
                               zeta_scale=None, 
                               zeta_min=None, 
                               beta=None, 
                               beta_min=None):
        """
        End-to-end pipeline to reconstruct CryoEM images.
        
        Args:
            file_path: Path to CryoEM dataset file (.pt or .mrcs)
            image_ids: List of image IDs to reconstruct (default: [0])
            num_masks: Number of masks to generate
            mask_prob: Probability for binary mask generation
            mask_type: Type of mask
            noise_level: Standard deviation of noise to add to measurements
            num_timesteps: Number of diffusion timesteps
            zeta_scale: Final scale factor for the step size
            zeta_min: Initial scale factor for the step size
            beta: Final momentum factor
            beta_min: Initial momentum factor
            
        Returns:
            Tuple of (reconstructed_images, original_images, metrics)
        """
        if self.unet is None or self.scheduler is None:
            raise ValueError("DDPM model not loaded. Call load_model() first or initialize with a model_path.")
        
        # Default to single image reconstruction if no IDs provided
        if image_ids is None:
            image_ids = [0]
        
        batch_size = len(image_ids)
        
        # 1. Generate masks
        masks = self.generate_masks(num_masks, mask_prob, mask_type)
        
        # 2. Load CryoEM images
        print(f"Loading CryoEM images (IDs: {image_ids})...")
        original_images = load_cryoem_batch(
            file_path, 
            image_ids, 
            self.img_size, 
            self.device
        )
        
        # 3. Get measurements
        print("Getting measurements from original images...")
        target_measurements = self.get_measurements(original_images, masks)
        
        # 4. Save all measurements if verbose mode is enabled
        self.save_measurements(target_measurements, batch_size)
        
        # 5. Plot measurements and original images
        plot_measurements_and_original(
            original_images, 
            target_measurements, 
            masks, 
            self.result_dir
        )
        
        # 6. Add noise to measurements if specified
        if noise_level > 0:
            print(f"Adding Gaussian noise with sigma={noise_level} to measurements...")
            target_measurements = add_gaussian_noise(target_measurements, noise_level)
        
        # 7. Reconstruct images
        reconstructed_images = self.reconstruct_image(
            target_measurements,
            masks,
            batch_size,
            num_timesteps,
            zeta_scale,
            zeta_min,
            beta,
            beta_min,
            image_ids
        )
        
        # 8. Evaluate reconstruction
        experiment_params = {
            'block_size': self.block_size,
            'num_masks': num_masks,
            'mask_type': mask_type,
            'noise_level': noise_level
        }
        
        metrics = self.evaluate_reconstruction(
            original_images,
            reconstructed_images,
            target_measurements,
            masks,
            image_ids
        )
        
        # 9. Save reconstructed images
        for i, image_id in enumerate(image_ids):
            # Convert from [-1, 1] to [0, 1] for visualization
            recon_vis = (reconstructed_images[i:i+1] + 1) / 2
            
            # Save as PNG
            plt.figure(figsize=(8, 8), frameon=False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            if self.in_channels == 1:
                plt.imshow(recon_vis[0, 0].cpu().numpy(), cmap='gray')
            else:
                plt.imshow(recon_vis[0, :min(3, self.in_channels)].permute(1, 2, 0).cpu().numpy())
            
            plt.axis('off')
            plt.savefig(f"{self.result_dir}/reconstructed_image_{image_id}.png", bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Save the raw reconstructed tensor (in [-1,1] range) as .pt file
            torch.save(reconstructed_images[i:i+1].cpu(), f"{self.result_dir}/reconstruction_raw_image_{image_id}.pt")
        
        print(f"\nReconstruction completed for {batch_size} images.")
        print(f"Results saved in '{self.result_dir}'")
        
        return reconstructed_images, original_images, metrics 