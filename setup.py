from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cryogen",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@gmail.com",
    description="CryoEM Image Reconstruction with Diffusion Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
        "diffusers>=0.11.0",
        "scikit-image>=0.18.0",
        "lpips>=0.1.4",
        "mrcfile>=1.4.0",
        "imageio>=2.9.0",
        "accelerate>=1.7.0",
        "pandas>=2.2.3",
    ],
    entry_points={
        "console_scripts": [
            "cryogen=CryoGEN.cli:main",
        ],
    },
)
