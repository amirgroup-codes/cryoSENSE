# CryoGEN: CryoEM Image Reconstruction with Diffusion Models

CryoGEN is a Python package for reconstructing CryoEM images using diffusion models. It implements the CryoGEN algorithm, which uses diffusion-based priors for enhanced image reconstruction from limited measurements.

## Features

- Efficient measurement and reconstruction of CryoEM images
- Support for various mask types (random binary, random Gaussian, checkerboard, moiré patterns)
- Batch processing of multiple images
- Comprehensive reconstruction quality evaluation
- Easy-to-use Python API and command-line interface
- Verbose mode with detailed visualizations including GIF animations of the diffusion process

## Installation

### From PyPI (recommended)

```bash
pip install cryogen
```

### From Source

```bash
git clone https://github.com/dsene/CryoGEN.git
cd CryoGEN
pip install -e .
```

## Quick Start

### Command-line Interface

```bash
# Basic usage
cryogen --model /path/to/ddpm/model --cryoem_path /path/to/cryoem/data.pt --start_id 0 --end_id 10

# Advanced options
cryogen --model /path/to/ddpm/model \
        --cryoem_path /path/to/cryoem/data.pt \
        --block_size 16 \
        --num_masks 50 \
        --mask_type random_binary \
        --zeta_scale 0.1 \
        --zeta_min 0.01 \
        --num_timesteps 1000 \
        --beta 0.9 \
        --beta_min 0.1 \
        --start_id 0 \
        --end_id 10 \
        --batch_size 4 \
        --noise_level 0.05 \
        --result_dir ./reconstruction_results

# With verbose output and visualizations
cryogen --model /path/to/ddpm/model \
        --cryoem_path /path/to/cryoem/data.pt \
        --start_id 0 \
        --end_id 0 \
        --result_dir ./verbose_results \
        --verbose
```

### Python API

```python
from CryoGEN.main import CryoGEN

# Initialize CryoGEN with a pretrained DDPM model
cryogen = CryoGEN(
    model_path="/path/to/ddpm/model",
    block_size=16,
    result_dir="./results",
    verbose=True  # Enable detailed visualizations
)

# Reconstruct images from a CryoEM dataset
reconstructed_images, original_images, metrics = cryogen.reconstruct_from_cryoem(
    file_path="/path/to/cryoem/data.pt",
    image_ids=[0, 1, 2],  # Process images with these IDs
    num_masks=30,
    mask_type="random_binary",
    num_timesteps=1000,
    zeta_scale=0.1,
    beta=0.9
)

# Access reconstruction metrics
for metric in metrics:
    print(f"Image ID: {metric['image_id']}, PSNR: {metric['PSNR']}, SSIM: {metric['SSIM']}")
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to the pretrained DDPM model | (required) |
| `--cryoem_path` | Path to the CryoEM dataset file | (required) |
| `--block_size` | Block size for downsampling | 4 |
| `--num_masks` | Number of binary masks to use | 30 |
| `--mask_prob` | Probability for binary mask generation | 0.5 |
| `--mask_type` | Type of mask to use (random_binary, random_gaussian, checkerboard, moire) | random_binary |
| `--zeta_scale` | Scale factor for the gradient step size | 0.1 |
| `--zeta_min` | Initial scale factor for the gradient step size | 0.01 |
| `--num_timesteps` | Number of diffusion timesteps | 1000 |
| `--beta` | Final momentum factor for updates | 0.9 |
| `--beta_min` | Initial momentum factor for updates | 0.1 |
| `--start_id` | Starting index of images to process | 0 |
| `--end_id` | Ending index of images to process | 0 |
| `--batch_size` | Number of images to process in each batch | 1 |
| `--noise_level` | Gaussian noise standard deviation for measurements | 0.0 |
| `--result_dir` | Directory to save results | results |
| `--device` | Device to use (cuda or cpu) | cuda |
| `--verbose` | Enable verbose mode with detailed visualizations | False |

## Verbose Mode Output

When the `--verbose` flag is enabled, CryoGEN will generate and save:

1. All binary masks used for measurements
2. All measurement images for each mask
3. Reconstructed images in PNG format
4. Comparison images showing original, reconstructed, and error maps
5. A GIF animation showing the diffusion process from t=1000 to t=1

Regardless of whether verbose mode is enabled, CryoGEN always saves:
- Raw reconstructed image tensors (.pt files)
- Reconstruction metrics in CSV format

## Supported Data Formats

- PyTorch tensor files (`.pt`) containing CryoEM images
- MRC files (`.mrcs`) containing CryoEM images

## Requirements

- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- diffusers 0.11.0 or higher
- CUDA-capable GPU (recommended)

## Citation

If you use CryoGEN in your research, please cite:

```bibtex
@article{cryogen2023,
  title={CryoGEN: Cryo-EM Image Reconstruction with Diffusion Models},
  author={Senejohnny, Danial and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## License

MIT License
