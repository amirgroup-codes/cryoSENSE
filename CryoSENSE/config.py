"""
Configuration utilities for CryoSENSE.
"""

import os
import json
import pkg_resources

def get_config_path(block_size=None, config_name=None):
    """
    Get the path to the configuration file.
    
    Args:
        block_size: Block size for downsampling (optional)
        config_name: Name of the configuration file without extension (optional)
        
    Returns:
        Path to the configuration file
    """
    if config_name:
        config_file = f"{config_name}.json"
    elif block_size:
        # Try to find a specific config for this block size
        config_file = f"block_size_{block_size}.json"
    else:
        # Default config
        config_file = "default.json"
    
    # Check if the config exists in the installed package
    try:
        config_path = pkg_resources.resource_filename('CryoSENSE', f'../configs/{config_file}')
        if os.path.exists(config_path):
            return config_path
    except (ImportError, FileNotFoundError):
        pass
    
    # Try in the local configs directory
    local_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', config_file)
    if os.path.exists(local_config_path):
        return local_config_path
    
    # If no specific config and block_size is provided, use a fallback based on block size
    if not config_name and block_size:
        if block_size <= 16:
            # For small block sizes, use block_size_16.json
            fallback_file = "block_size_16.json"
        else:
            # For large block sizes, use block_size_32.json
            fallback_file = "block_size_32.json"
        
        # Try to find the fallback config
        try:
            config_path = pkg_resources.resource_filename('CryoSENSE', f'../configs/{fallback_file}')
            if os.path.exists(config_path):
                return config_path
        except (ImportError, FileNotFoundError):
            pass
        
        # Local fallback
        local_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', fallback_file)
        if os.path.exists(local_config_path):
            return local_config_path
    
    # Try default config as final fallback
    if not config_name and not block_size:
        default_file = "default.json"
        try:
            config_path = pkg_resources.resource_filename('CryoSENSE', f'../configs/{default_file}')
            if os.path.exists(config_path):
                return config_path
        except (ImportError, FileNotFoundError):
            pass
        
        # Local default
        local_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', default_file)
        if os.path.exists(local_config_path):
            return local_config_path
    
    return None

def load_config(block_size=None, config_name=None):
    """
    Load configuration parameters.
    
    Args:
        block_size: Block size for downsampling (optional)
        config_name: Name of the configuration file without extension (optional)
        
    Returns:
        Dictionary of configuration parameters
    """
    config_path = get_config_path(block_size, config_name)
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if block_size:
            print(f"Loaded configuration for block size {block_size} from {config_path}")
            if 'block_size' in config and config['block_size'] != block_size:
                print(f"Warning: Requested block size {block_size} but loaded configuration for block size {config['block_size']}")
        else:
            print(f"Loaded configuration from {config_path}")
        
        return config
    else:
        # Default configuration if no file is found
        if block_size:
            print(f"No configuration file found for block size {block_size}. Using default parameters.")
            
            if block_size <= 16:
                zeta_scale = 1.0
            else:
                zeta_scale = 10.0
                
            return {
                "block_size": block_size,
                "zeta_scale": zeta_scale,
                "zeta_min": 1e-2,
                "beta": 0.9,
                "beta_min": 0.1,
                "description": "Default parameters"
            }
        else:
            print("No configuration file found. Using hardcoded default parameters.")
            return {
                "block_size": 4,
                "zeta_scale": 1.0,
                "zeta_min": 1e-2,
                "beta": 0.9,
                "beta_min": 0.1,
                "num_masks": 30,
                "mask_prob": 0.5,
                "mask_type": "random_binary",
                "num_timesteps": 1000,
                "description": "Hardcoded default parameters"
            }

def get_recommended_params(block_size):
    """
    Get recommended parameters for a given block size.
    
    Args:
        block_size: Block size for downsampling
        
    Returns:
        Tuple of (zeta_scale, zeta_min, beta, beta_min)
    """
    config = load_config(block_size)
    
    return (
        config.get("zeta_scale", 1.0 if block_size <= 16 else 10.0),
        config.get("zeta_min", 1e-2),
        config.get("beta", 0.9),
        config.get("beta_min", 0.1)
    )

def load_default_config():
    """
    Load the default configuration for CryoSENSE.
    
    Returns:
        Dictionary of default configuration parameters
    """
    return load_config(config_name="default") 