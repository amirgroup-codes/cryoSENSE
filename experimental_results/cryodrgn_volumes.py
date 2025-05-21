import os
import sys
import pickle
import numpy as np
from cryodrgn import analysis

# Data directory containing all the required files
DATA_DIR = 'cryodrgn_data'

# Output directory
OUTPUT_DIR = 'cryodrgn_volumes'

# Particles to decode
PARTICLES_TO_DECODE = [4046, 352, 4867, 15117]

# Volume generation parameters
APIX = 3.275  # Pixel size in Angstroms
FLIP = True   # Flip handedness of output volumes
DEVICE = 0    # GPU device index

# Define dataset paths
DATASETS = {
    'cryogen': {
        'z_file': os.path.join(DATA_DIR, 'cryogen_z.99.pkl'),
        'weights_file': os.path.join(DATA_DIR, 'cryogen_weights.99.pkl'),
        'config_file': os.path.join(DATA_DIR, 'cryogen_config.yaml'),
        'outdir': os.path.join(OUTPUT_DIR, 'cryogen_volumes')
    },
    'original': {
        'z_file': os.path.join(DATA_DIR, 'original_z.99.pkl'),
        'weights_file': os.path.join(DATA_DIR, 'original_weights.99.pkl'),
        'config_file': os.path.join(DATA_DIR, 'original_config.yaml'),
        'outdir': os.path.join(OUTPUT_DIR, 'original_volumes')
    },
    'lowres': {
        'z_file': os.path.join(DATA_DIR, 'lowres_z.99.pkl'),
        'weights_file': os.path.join(DATA_DIR, 'lowres_weights.99.pkl'),
        'config_file': os.path.join(DATA_DIR, 'lowres_config.yaml'),
        'outdir': os.path.join(OUTPUT_DIR, 'lowres_volumes')
    }
}

def check_files_exist():
    """Check if all required files exist in the data directory"""
    missing_files = []
    for dataset_name, paths in DATASETS.items():
        for key in ['z_file', 'weights_file', 'config_file']:
            if not os.path.exists(paths[key]):
                missing_files.append(paths[key])
    
    if missing_files:
        print(f"ERROR: The following files are missing in the {DATA_DIR} directory:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def decode_particle_volumes(dataset_name):
    """Decode specific particle volumes for a given dataset"""
    dataset = DATASETS[dataset_name]
    
    # Create output directory
    os.makedirs(dataset['outdir'], exist_ok=True)
    
    try:
        # Load z values
        print(f"Loading z values from {dataset['z_file']}")
        with open(dataset['z_file'], 'rb') as f:
            z_data = pickle.load(f)
        
        # Extract z values for specified particles
        z_values = z_data[PARTICLES_TO_DECODE]
        print(f"Extracting z values for {len(PARTICLES_TO_DECODE)} particles")
        
        # Save z values to temporary file
        z_temp_file = os.path.join(dataset['outdir'], "z_values_to_decode.txt")
        np.savetxt(z_temp_file, z_values)
        
        # Generate volumes with the specified parameters
        print(f"Generating volumes for {dataset_name} with Apix={APIX}, flip={FLIP}...")
        analysis.gen_volumes(
            dataset['weights_file'],
            dataset['config_file'],
            z_temp_file,
            dataset['outdir'],
            Apix=APIX,
            flip=FLIP,
            device=DEVICE
        )
        
        # Clean up temporary file
        os.remove(z_temp_file)
        
        # Rename volumes to include particle IDs
        for i, particle_id in enumerate(PARTICLES_TO_DECODE):
            old_name = os.path.join(dataset['outdir'], f"vol_{i:03d}.mrc")
            new_name = os.path.join(dataset['outdir'], f"{dataset_name}_particle_{particle_id}.mrc")
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
                print(f"Renamed volume for particle {particle_id}")
            else:
                print(f"Warning: Could not find volume for particle {particle_id}")
        
        # Save particle indices for reference
        np.savetxt(os.path.join(dataset['outdir'], "particle_indices.txt"), PARTICLES_TO_DECODE, fmt="%d")
        print(f"Volumes for {dataset_name} saved to {dataset['outdir']}")
        return True
    except Exception as e:
        print(f"Error processing {dataset_name} dataset: {e}")
        return False

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if required files exist
    if not check_files_exist():
        print("Required files are missing. Please check the error messages above.")
        return
    
    # Process each dataset
    successful_datasets = []
    for dataset_name in DATASETS.keys():
        print(f"\nProcessing {dataset_name} dataset...")
        success = decode_particle_volumes(dataset_name)
        if success:
            successful_datasets.append(dataset_name)
    
    # Create a summary report if at least one dataset was processed successfully
    if successful_datasets:
        with open(os.path.join(OUTPUT_DIR, "volume_summary.txt"), 'w') as f:
            f.write("Figure 3 Volume Summary\n")
            f.write("=====================\n\n")
            f.write(f"Generated volumes for {len(PARTICLES_TO_DECODE)} particles from {len(successful_datasets)} datasets.\n\n")
            
            f.write("Parameters:\n")
            f.write(f"  - Apix: {APIX}\n")
            f.write(f"  - Flip handedness: {FLIP}\n")
            f.write(f"  - GPU device: {DEVICE}\n\n")
            
            f.write("Particles:\n")
            for particle_id in PARTICLES_TO_DECODE:
                f.write(f"  - Particle {particle_id}\n")
            
            f.write("\nDatasets:\n")
            for dataset_name in successful_datasets:
                f.write(f"  - {dataset_name}\n")
            
            f.write("\nOutput files:\n")
            for dataset_name in successful_datasets:
                outdir = DATASETS[dataset_name]['outdir']
                f.write(f"  {dataset_name} volumes in: {outdir}\n")
                for particle_id in PARTICLES_TO_DECODE:
                    vol_file = f"{dataset_name}_particle_{particle_id}.mrc"
                    f.write(f"    - {vol_file}\n")
        
        print("\nAll volumes generated successfully.")
        print(f"Volumes are available in dataset-specific subdirectories of {OUTPUT_DIR}/")
        print(f"Volumes were generated with Apix={APIX} and flip={FLIP}")
    else:
        print("\nFailed to generate any volumes. Please check the errors above.")

if __name__ == "__main__":
    main() 