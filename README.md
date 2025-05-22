# CryoGEN: CryoEM Image Reconstruction with Diffusion Models

<div align="center">
  <img src="logo.png" width="200" alt="CryoGEN Logo">
  
  ![Python](https://img.shields.io/badge/python-3.13.2-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)
  ![License](https://img.shields.io/badge/license-MIT-green.svg)
  ![Status](https://img.shields.io/badge/status-active-brightgreen.svg)
</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features) 
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Demonstration](#-demonstration)
- [Examples](#-examples)
- [Experiments](#-experiments)
- [Pre-trained Models](#-pre-trained-models)
- [Datasets](#-datasets)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Requirements](#-requirements)
- [Comparative Methods](#-comparative-methods)
- [Contributing](#-contributing)

---

## 🔬 Overview

Deep generative models have recently shown promise as priors for solving inverse problems, enabling image recovery without reliance on sparsity assumptions on a pre-defined basis. Diffusion models, in particular, have enabled super-resolution, inpainting, and deblurring of natural images by learning data distributions over low-dimensional image manifolds. However, scientific imaging modalities, such as cryo-electron microscopy (cryo-EM), demand accurate rather than merely perceptually plausible reconstructions, as visually minor errors can lead to incorrect structural interpretations.

**CryoGEN** is a generative model for solving the cryo-EM inverse problem of reconstructing biologically accurate high-resolution images from compressed, low-resolution linear measurements. CryoGEN couples an unconditional denoising diffusion probabilistic model (DDPM) trained on cryo-EM data with Nesterov-accelerated gradients to steer the reverse diffusion toward a solution consistent with the compressed measurements.

## ✨ Key Features

- **High-Resolution Recovery**: Enables recovery from inputs up to **32× lower resolution**
- **Structural Preservation**: Maintains critical structural information for downstream analysis
- **Atomic Model Building**: Supports atomic model building and conformational heterogeneity analysis
- **Flexible Configuration**: Multiple block sizes and mask types for different compression levels
- **Pre-trained Models**: Ready-to-use models for various datasets
- **Comprehensive API**: Both command-line and Python API interfaces

---

## 🚀 Installation

### From PyPI (Recommended)

> **Note**: This option will be enabled in the camera-ready version.

```bash
pip install cryogen
```

### From Source

```bash
git clone X
cd CryoGEN
pip install -e .
```

---

## ⚡ Quick Start

### Command-line Interface

#### Basic Usage
```bash
cryogen --model /path/to/ddpm/model \
        --cryoem_path /path/to/cryoem/data.pt \
        --start_id 0 \
        --end_id 10
```

#### Advanced Options
```bash
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
```

#### Verbose Mode with Visualizations
```bash
cryogen --model /path/to/ddpm/model \
        --cryoem_path /path/to/cryoem/data.pt \
        --start_id 0 \
        --end_id 0 \
        --result_dir ./verbose_results \
        --verbose
```

#### Using Configuration Files
```bash
cryogen --model /path/to/ddpm/model \
        --cryoem_path /path/to/cryoem/data.pt \
        --block_size 32 \
        --use_config \
        --start_id 0 \
        --end_id 10 \
        --result_dir ./config_results
```

### Python API

```python
from CryoGEN.main import CryoGEN

# Initialize CryoGEN with a pretrained DDPM model
cryogen = CryoGEN(
    model_path="/path/to/ddpm/model",
    block_size=16,
    result_dir="./results",
    verbose=True,  # Enable detailed visualizations
    use_config=True  # Use recommended parameters from configuration files
)

# Reconstruct images from a CryoEM dataset
reconstructed_images, original_images, metrics = cryogen.reconstruct_from_cryoem(
    file_path="/path/to/cryoem/data.pt",
    image_ids=[0, 1, 2],  # Process images with these IDs
    num_masks=30,
    mask_type="random_binary",
    num_timesteps=1000
    # Parameters like zeta_scale and beta will be loaded from the configuration file
)

# Access reconstruction metrics
for metric in metrics:
    print(f"Image ID: {metric['image_id']}, PSNR: {metric['PSNR']}, SSIM: {metric['SSIM']}")
```

---

## 🎯 Demonstration

CryoGEN enables high-quality reconstruction across different downsampling levels. Below we demonstrate two configurations:

### Low Compression (Block Size 2, 4 Masks)

With minimal compression (block size 2), CryoGEN can reconstruct high-quality images using just **4 masks**:

<div align="center">
  <img src="results/block2_4masks/reconstruction_comparison_image_0.png" width="600" alt="Low compression reconstruction">
  <p><em>Reconstruction comparison with minimal compression</em></p>
</div>

The reconstruction is guided by compressed measurements:

<div align="center">
  <img src="results/block2_4masks/measurements/img0_measurement_0.png" width="200" alt="Measurement visualization">
  <p><em>Compressed measurement visualization</em></p>
</div>

The diffusion process gradually builds the image from random noise:

<div align="center">
  <img src="results/block2_4masks/diffusion_process_img0.gif" width="300" alt="Diffusion process">
</div>

### High Compression (Block Size 32, 1024 Masks)

Even with extreme compression (block size 32), CryoGEN reconstructs detailed protein structures using **1024 masks**:

<div align="center">
  <img src="results/block32_1024masks/reconstruction_comparison_image_0.png" width="600" alt="High compression reconstruction">
  <p><em>Reconstruction comparison with extreme compression</em></p>
</div>

The reconstruction is guided by highly compressed measurements:

<div align="center">
  <img src="results/block32_1024masks/measurements/img0_measurement_0.png" width="200" alt="Highly compressed measurement">
  <p><em>Highly compressed measurement visualization</em></p>
</div>

The diffusion process gradually builds the image from random noise:

<div align="center">
  <img src="results/block32_1024masks/diffusion_process_img0.gif" width="300" alt="Progressive diffusion">
</div>

---

## 📋 Examples

CryoGEN includes ready-to-use example scripts for quick testing and demonstration:

### Python Example
```bash
python examples/simple_example.py
```

### Bash Example
```bash
bash examples/simple_example.sh
```

### Example Features

- **🔄 No Manual Downloads**: Sample image included in the `data/` directory
- **🤖 Pre-trained Model**: Uses anonymously uploaded DDPM model from HuggingFace
- **⚙️ Configurable Parameters**: Block size adjustable to {2, 4, 8, 16, 32}
- **📊 Results Visualization**: Verbose output with detailed visualizations

The scripts demonstrate reconstruction with different parameter combinations:
1. **Block size 32** with 1024 masks
2. **Block size 2** with 4 masks

---

## 🧪 Experiments

CryoGEN provides advanced scripts for comprehensive experiments and batch processing:

### Grid Search Experiments

```bash
python scripts/run_experiments.py
```

The `run_experiments.py` script reproduces results from the paper, including LPIPS and SSIM scores for CryoGEN reconstructions across five downsampling levels (2x-32x).

**Features:**
- **📊 Grid Search**: Multiple parameters (block sizes, masks, types, noise levels)
- **🔍 Analysis**: Identifies optimal configurations automatically
- **📈 Comprehensive Reports**: Generates metrics (PSNR, SSIM, LPIPS)
- **🖥️ Multi-GPU Support**: Parallel processing across multiple GPUs

### Batch Reconstruction

```bash
python scripts/reconstruct_all_images.py
```

The `reconstruct_all_images.py` script processes entire datasets using optimal configurations.

**Features:**
- **⚡ Efficient Processing**: Handles entire datasets
- **🔄 Auto Distribution**: Work distributed across available GPUs
- **⚙️ Fixed Configuration**: Uses optimal parameters for reconstruction
- **📊 Comprehensive Metrics**: Generates metrics for all processed images
- **📁 Format Support**: PyTorch tensors and MRC files

---

## 🤖 Pre-trained Models

The following DDPM models are available on HuggingFace and can be used directly with CryoGEN:

| Model | Resolution | Dataset | Description |
|-------|------------|---------|-------------|
| `anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128` | 128×128 | EMPIAR10076 | General purpose CryoEM model |
| `anonymousneurips008/empiar11526-ddpm-ema-cryoem-128x128` | 128×128 | EMPIAR11526 | Specialized for dataset 11526 |
| `anonymousneurips008/empiar10166-ddpm-ema-cryoem-128x128` | 128×128 | EMPIAR10166 | Optimized for dataset 10166 |
| `anonymousneurips008/empiar10786-ddpm-ema-cryoem-128x128` | 128×128 | EMPIAR10786 | Trained on dataset 10786 |
| `anonymousneurips008/empiar10648-ddpm-cryoem-256x256` | 256×256 | EMPIAR10648 | High-resolution model |

### Usage Example
```bash
cryogen --model anonymousneurips008/empiar10076-ddpm-ema-cryoem-128x128 \
        --cryoem_path /path/to/data
```

---

## 📊 Datasets

The following datasets are available on HuggingFace:

| Dataset | Type | Description |
|---------|------|-------------|
| `anonymousneurips008/3D_Volumes_EMPIAR10076` | 3D Volumes | 3D volumes of EMPIAR10076 |
| `anonymousneurips008/3D_Volumes_EMPIAR10648` | 3D Volumes | 3D volumes of EMPIAR10648 |
| `anonymousneurips008/EMPIAR10076_128x128` | 2D Images | EMPIAR10076 128×128 Images |
| `anonymousneurips008/EMPIAR11526_128x128` | 2D Images | EMPIAR11526 128×128 Images |
| `anonymousneurips008/EMPIAR10166_128x128` | 2D Images | EMPIAR10166 128×128 Images |
| `anonymousneurips008/EMPIAR10786_128x128` | 2D Images | EMPIAR10786 128×128 Images |
| `anonymousneurips008/EMPIAR10648_256x256` | 2D Images | EMPIAR10648 256×256 Images |

---

## ⚙️ Configuration

### Command-line Options

| Option | Description | Default | Type |
|--------|-------------|---------|------|
| `--model` | Path to the pretrained DDPM model | **(required)** | str |
| `--cryoem_path` | Path to the CryoEM dataset file | **(required)** | str |
| `--block_size` | Block size for downsampling | `4` | int |
| `--num_masks` | Number of binary masks to use | `30` | int |
| `--mask_prob` | Probability for binary mask generation | `0.5` | float |
| `--mask_type` | Type of mask (random_binary, random_gaussian, checkerboard) | `random_binary` | str |
| `--zeta_scale` | Scale factor for the gradient step size | *(from config)* | float |
| `--zeta_min` | Initial scale factor for the gradient step size | *(from config)* | float |
| `--num_timesteps` | Number of diffusion timesteps | `1000` | int |
| `--beta` | Final momentum factor for updates | *(from config)* | float |
| `--beta_min` | Initial momentum factor for updates | *(from config)* | float |
| `--start_id` | Starting index of images to process | `0` | int |
| `--end_id` | Ending index of images to process | `0` | int |
| `--batch_size` | Number of images to process in each batch | `1` | int |
| `--noise_level` | Gaussian noise standard deviation for measurements | `0.0` | float |
| `--result_dir` | Directory to save results | `results` | str |
| `--device` | Device to use (cuda or cpu) | `cuda` | str |
| `--verbose` | Enable verbose mode with detailed visualizations | `False` | bool |
| `--use_config` | Use recommended configuration parameters | `False` | bool |

### Recommended Parameters

CryoGEN includes configuration files with optimized parameters based on block size:

| Block Size | zeta_scale | zeta_min | beta | beta_min | Use Case |
|------------|------------|----------|------|----------|----------|
| 2, 4, 8, 16 | `1.0` | `1e-2` | `0.9` | `0.1` | Low to medium compression |
| 32, 64 | `10.0` | `1e-2` | `0.9` | `0.1` | High compression |

**To use recommended configurations:**
1. Pass `--use_config` on the command line, or
2. Set `use_config=True` when creating a CryoGEN instance

You can override any specific parameter by explicitly providing it.

### Verbose Mode Output

When `--verbose` is enabled, CryoGEN generates:

- ✅ All binary masks used for measurements
- ✅ All measurement images for each mask  
- ✅ Reconstructed images in PNG format
- ✅ Comparison images (original, reconstructed, error maps)
- ✅ GIF animation of the diffusion process (t=1000 to t=1)

**Always saved regardless of verbose mode:**
- Raw reconstructed image tensors (`.pt` files)
- Reconstruction metrics in CSV format

---

## 🔧 API Reference

### Supported Data Formats

- **PyTorch tensor files** (`.pt`) containing CryoEM images
- **MRC files** (`.mrcs`) containing CryoEM images

### Python API Classes

```python
# Main CryoGEN class
class CryoGEN:
    def __init__(self, model_path, block_size=4, result_dir="./results", 
                 verbose=False, use_config=False)
    
    def reconstruct_from_cryoem(self, file_path, image_ids, num_masks=30, 
                               mask_type="random_binary", num_timesteps=1000)
```

---

## 📦 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | `3.13.2` | Core runtime |
| PyTorch | `2.7.0` | Deep learning framework |
| diffusers | `0.33.1` | Diffusion model utilities |
| accelerate | `1.7.0` | Training acceleration |
| numpy | `2.2.6` | Numerical computing |
| scikit-image | `0.25.2` | Image processing |
| lpips | `0.1.4` | Perceptual metrics |
| mrcfile | `1.5.4` | MRC file handling |
| matplotlib | `3.10.3` | Visualization |
| imageio | `2.37.0` | Image I/O |

**Hardware Requirements:**
- 🖥️ **CUDA-capable GPU** (recommended)
- 💾 **Minimum 8GB RAM**
- 💽 **10GB free disk space**

---

## 🔬 Comparative Methods

You can find the code for running comparative method reconstruction experiments in the `comparative_methods/` directory. For detailed instructions, please refer to the `comparative_methods/README.md` file.

The scripts and data for analyzing and visualizing experimental results are located in the `experimental_results/` directory. For more details, refer to the `experimental_results/README.md` file.

---

## 🤝 Contributing

We welcome contributions to CryoGEN! Please feel free to:

1. 🐛 **Report bugs** by opening an issue
2. 💡 **Suggest features** through feature requests  
3. 🔧 **Submit pull requests** for bug fixes or enhancements
4. 📖 **Improve documentation** with clearer examples

---

<div align="center">
  <p><strong>CryoGEN</strong> - Advancing CryoEM Image Reconstruction with Diffusion Models</p>
  <p>Built with ❤️ for the scientific community</p>
</div>
