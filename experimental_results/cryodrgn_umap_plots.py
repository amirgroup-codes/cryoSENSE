# Set thread limits
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import umap
from sklearn.mixture import GaussianMixture

# Data directory containing all the required files
DATA_DIR = 'cryodrgn_data'

# Data files in the cryodrgn_plots_data directory
Z_CRYOGEN_FILE = os.path.join(DATA_DIR, 'cryogen_z.99.pkl')
Z_ORIGINAL_FILE = os.path.join(DATA_DIR, 'original_z.99.pkl')
Z_LOWRES_FILE = os.path.join(DATA_DIR, 'lowres_z.99.pkl')

# GMM and UMAP results in the cryodrgn_plots_data directory
CRYOGEN_GMM_FILE = os.path.join(DATA_DIR, 'cryogen_gmm_results.pkl')
ORIGINAL_GMM_FILE = os.path.join(DATA_DIR, 'original_gmm_results.pkl')
LOWRES_GMM_FILE = os.path.join(DATA_DIR, 'lowres_gmm_results.pkl')

CRYOGEN_UMAP_FILE = os.path.join(DATA_DIR, 'cryogen_umap_results.pkl')
ORIGINAL_UMAP_FILE = os.path.join(DATA_DIR, 'original_umap_results.pkl')
LOWRES_UMAP_FILE = os.path.join(DATA_DIR, 'lowres_umap_results.pkl')

# Output directory
OUTPUT_DIR = 'cryodrgn_plots'

# Particles to label in the plots
PARTICLES_TO_LABEL = [15761, 8973, 4051, 11409]

# GMM and UMAP parameters
N_COMPONENTS = 4
RANDOM_STATE = 42

def load_z_value(file_path):
    """Load z values from a specific file path"""
    print(f"Loading z values from {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_gmm_results(file_path):
    """Load GMM results from a file"""
    print(f"Loading GMM results from {file_path}")
    try:
        with open(file_path, 'rb') as f:
            gmm_data = pickle.load(f)
            return gmm_data['gmm_labels']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading GMM results: {e}")
        return None

def load_umap_results(file_path):
    """Load UMAP results from a file"""
    print(f"Loading UMAP results from {file_path}")
    try:
        with open(file_path, 'rb') as f:
            umap_data = pickle.load(f)
            return umap_data['umap_embedding']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading UMAP results: {e}")
        return None

def compute_umap(z_values, random_state=RANDOM_STATE):
    """Compute UMAP embedding from z values"""
    print(f"Computing UMAP for {z_values.shape[0]} particles with z dimension {z_values.shape[1]}")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=random_state
    )
    return reducer.fit_transform(z_values)

def compute_gmm(data, n_components=N_COMPONENTS, random_state=RANDOM_STATE):
    """Compute Gaussian Mixture Model clustering on UMAP embedding"""
    print(f"Computing GMM with {n_components} components on UMAP embedding")
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        max_iter=100,
        n_init=5
    )
    gmm.fit(data)
    return gmm.predict(data)

def print_cluster_correlations(gmm_labels1, gmm_labels2):
    """Print cluster correlations between two sets of cluster labels and save to file"""
    print("\nCluster Correlation Analysis:")
    print("-----------------------------")
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(gmm_labels1, gmm_labels2)
    
    # Calculate percentages
    row_sums = conf_matrix.sum(axis=1)
    conf_matrix_percent = conf_matrix / row_sums[:, np.newaxis] * 100
    
    # Prepare output string for both console and file
    output = []
    output.append("Cluster Correlation Analysis:")
    output.append("-----------------------------")
    output.append("\nPercentage of Cryogen dataset clusters in Original dataset clusters:")
    
    for i in range(len(np.unique(gmm_labels1))):
        output.append(f"Cryogen Cluster {i}:")
        for j in range(len(np.unique(gmm_labels2))):
            output.append(f"  → Original Cluster {j}: {conf_matrix_percent[i, j]:.1f}%")
    
    output.append("\nStrongest cluster correspondences:")
    for i in range(len(np.unique(gmm_labels1))):
        best_match = np.argmax(conf_matrix_percent[i, :])
        output.append(f"Cryogen Cluster {i} → Original Cluster {best_match} ({conf_matrix_percent[i, best_match]:.1f}%)")
    
    # Print to console
    for line in output:
        print(line)
    
    # Save to file
    output_dir = "cryodrgn_volumes"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cluster_correlations.txt")
    
    print(f"\nSaving cluster correlations to {output_file}")
    with open(output_file, 'w') as f:
        f.write("\n".join(output))

def create_paper_hexbin(umap_data, gmm_labels, title, filename, labeled_indices=None, 
                       figsize=(14, 12), dpi=300, use_gray_only=False, fixed_range=None):
    """Create a hexbin plot with consistent bin sizes and labeled particles."""
    # Calculate center and range
    center_x = (umap_data[:, 0].min() + umap_data[:, 0].max()) / 2
    center_y = (umap_data[:, 1].min() + umap_data[:, 1].max()) / 2
    
    if fixed_range is not None:
        max_range = fixed_range
    else:
        range_x = umap_data[:, 0].max() - umap_data[:, 0].min()
        range_y = umap_data[:, 1].max() - umap_data[:, 1].min()
        max_range = max(range_x, range_y) * 1.1  # Add 10% padding
    
    # Calculate plot limits
    x_min = center_x - max_range/2
    x_max = center_x + max_range/2
    y_min = center_y - max_range/2
    y_max = center_y + max_range/2
    
    # Define distinct colors for clusters
    distinct_colors = [
        (0.75, 0.10, 0.10),  # Darker Red
        (0.10, 0.55, 0.10),  # Darker Green
        (0.10, 0.35, 0.75),  # Darker Blue
        (0.50, 0.10, 0.55),  # Darker Purple
    ]
    
    # Create figure with GridSpec for main plot and histograms
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(20, 20, figure=fig, wspace=0.0, hspace=0.0)
    
    # Create axes: main plot, top histogram, right histogram
    ax_main = fig.add_subplot(gs[2:, :18], aspect='equal')  # Main UMAP plot
    ax_top = fig.add_subplot(gs[:2, :18])   # Top histogram
    ax_right = fig.add_subplot(gs[2:, 18:]) # Right histogram
    
    # Turn off all tick labels and ticks for the histograms
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    
    # Hide histogram spines
    for ax, spines in [(ax_top, ['top', 'right', 'left', 'bottom']), 
                       (ax_right, ['top', 'right', 'bottom', 'left'])]:
        for spine in spines:
            ax.spines[spine].set_visible(False)
    
    # Remove top and right spines from main plot
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # Set main plot limits
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    
    # Explicitly set histogram limits to match main plot
    ax_top.set_xlim(x_min, x_max)
    ax_right.set_ylim(y_min, y_max)
    
    if use_gray_only:
        # For gray-only hexbins
        hb = ax_main.hexbin(
            umap_data[:, 0], 
            umap_data[:, 1], 
            gridsize=50, 
            cmap='Greys', 
            alpha=0.7, 
            mincnt=5,
            extent=[x_min, x_max, y_min, y_max]
        )
    else:
        # First add a light background of all points
        ax_main.hexbin(
            umap_data[:, 0], 
            umap_data[:, 1], 
            gridsize=50, 
            cmap='Greys', 
            alpha=0.15, 
            mincnt=1,
            extent=[x_min, x_max, y_min, y_max]
        )
        
        # Plot each cluster with its own color
        for i in range(len(np.unique(gmm_labels))):
            mask = gmm_labels == i
            if np.sum(mask) > 0:
                # Get distinct color for this cluster
                cluster_color = distinct_colors[i % len(distinct_colors)]
                
                # Create a single-color colormap with darker colors
                dark_color = tuple(c * 0.9 for c in cluster_color) + (1.0,)
                cmap = LinearSegmentedColormap.from_list(f"cluster_{i}", 
                                                        [(1,1,1,0), dark_color],
                                                        N=256)
                
                # Plot hexbins for this cluster
                ax_main.hexbin(
                    umap_data[mask, 0], 
                    umap_data[mask, 1], 
                    gridsize=50, 
                    cmap=cmap, 
                    alpha=0.9,
                    mincnt=5,
                    extent=[x_min, x_max, y_min, y_max]
                )
    
    # Add labeled particles
    if labeled_indices:
        for idx in labeled_indices:
            if 0 <= idx < len(umap_data):
                x, y = umap_data[idx, 0], umap_data[idx, 1]
                # Use gray color for all circles
                circle = Circle(
                    (x, y),
                    radius=0.15,
                    facecolor='gray',
                    edgecolor='black',
                    linewidth=3.5,
                    alpha=1.0,
                    zorder=10
                )
                ax_main.add_patch(circle)
                
                # Add particle ID label
                ax_main.annotate(
                    f"Particle {idx}", 
                    (x, y),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    weight='bold',
                    color='black',
                    backgroundcolor='white',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
                    zorder=11
                )
    
    # Add histograms showing data density
    n_bins = 50
    
    # Use a percentile-based approach to determine visible data range
    x_percentile_low = np.percentile(umap_data[:, 0], 1)
    x_percentile_high = np.percentile(umap_data[:, 0], 99)
    y_percentile_low = np.percentile(umap_data[:, 1], 1)
    y_percentile_high = np.percentile(umap_data[:, 1], 99)
    
    # Create a mask for the "visible" points
    visible_mask = (
        (umap_data[:, 0] >= x_percentile_low) & 
        (umap_data[:, 0] <= x_percentile_high) & 
        (umap_data[:, 1] >= y_percentile_low) & 
        (umap_data[:, 1] <= y_percentile_high)
    )
    
    # Use filtered points for histograms
    visible_x = umap_data[visible_mask, 0]
    visible_y = umap_data[visible_mask, 1]
    
    # X-dimension histogram (top)
    ax_top.hist(visible_x, bins=n_bins, color='#888888', 
               range=(x_min, x_max), histtype='stepfilled', 
               edgecolor='none')
    
    # Y-dimension histogram (right)
    ax_right.hist(visible_y, bins=n_bins, color='#888888', 
                 range=(y_min, y_max), histtype='stepfilled', 
                 edgecolor='none', orientation='horizontal')
    
    # Add a legend with hexagons for clusters
    if not use_gray_only:
        handles = []
        for i in range(len(np.unique(gmm_labels))):
            color = distinct_colors[i % len(distinct_colors)]
            handle = RegularPolygon((0.5, 0.5), numVertices=6, radius=0.3,
                                facecolor=color, edgecolor=None)
            handles.append(handle)
        
        ax_main.legend(
            handles=handles, 
            labels=[f'Cluster {i}' for i in range(len(np.unique(gmm_labels)))],
            title='Clusters', 
            loc='upper right',
            fontsize=9
        )
    
    # Add title and labels
    fig.suptitle(title, fontsize=16, y=0.98)
    ax_main.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax_main.set_ylabel('UMAP Dimension 2', fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(f"{filename}.pdf")
    plt.close()
    print(f"Saved figure to {filename}.pdf")

def save_to_data_dir(data, filename):
    """Save data to the data directory"""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Saving data to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def main():
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory '{DATA_DIR}'...")
        os.makedirs(DATA_DIR, exist_ok=True)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # First try to load pre-computed GMM and UMAP results
    print("Checking for pre-computed UMAP and GMM results...")
    
    # Try to load pre-computed UMAP embeddings
    umap_embedding1 = load_umap_results(CRYOGEN_UMAP_FILE)
    umap_embedding2 = load_umap_results(ORIGINAL_UMAP_FILE)
    umap_embedding3 = load_umap_results(LOWRES_UMAP_FILE)
    
    # Try to load pre-computed GMM labels
    gmm_labels1 = load_gmm_results(CRYOGEN_GMM_FILE)
    gmm_labels2 = load_gmm_results(ORIGINAL_GMM_FILE)
    gmm_labels3 = load_gmm_results(LOWRES_GMM_FILE)
    
    # If any of the results couldn't be loaded, compute from scratch
    compute_from_scratch = False
    
    # Load z values if needed for computation
    if umap_embedding1 is None or umap_embedding2 is None or umap_embedding3 is None or \
       gmm_labels1 is None or gmm_labels2 is None or gmm_labels3 is None:
        compute_from_scratch = True
        print("Some pre-computed results couldn't be loaded. Loading z values...")
        z_cryogen = load_z_value(Z_CRYOGEN_FILE)
        z_original = load_z_value(Z_ORIGINAL_FILE)
        z_lowres = load_z_value(Z_LOWRES_FILE)
    
    # Compute or load UMAP embeddings from cache
    if compute_from_scratch:
        # First check if we're missing any UMAP embeddings
        if umap_embedding1 is None:
            print("Computing UMAP for cryosense dataset")
            umap_embedding1 = compute_umap(z_cryogen)
            # Save the UMAP embedding to the data directory
            save_to_data_dir({'umap_embedding': umap_embedding1}, 'cryogen_umap_results.pkl')
        
        if umap_embedding2 is None:
            print("Computing UMAP for original dataset")
            umap_embedding2 = compute_umap(z_original)
            # Save the UMAP embedding to the data directory
            save_to_data_dir({'umap_embedding': umap_embedding2}, 'original_umap_results.pkl')
        
        if umap_embedding3 is None:
            print("Computing UMAP for lowres dataset")
            umap_embedding3 = compute_umap(z_lowres)
            # Save the UMAP embedding to the data directory
            save_to_data_dir({'umap_embedding': umap_embedding3}, 'lowres_umap_results.pkl')
        
        # Now check if we're missing any GMM labels
        if gmm_labels1 is None:
            print("Computing GMM for cryosense dataset")
            # Run GMM on UMAP embedding
            gmm_labels1 = compute_gmm(umap_embedding1)
            # Save the GMM labels to the data directory
            save_to_data_dir({'gmm_labels': gmm_labels1}, 'cryogen_gmm_results.pkl')
        
        if gmm_labels2 is None:
            print("Computing GMM for original dataset")
            # Run GMM on UMAP embedding
            gmm_labels2 = compute_gmm(umap_embedding2)
            save_to_data_dir({'gmm_labels': gmm_labels2}, 'original_gmm_results.pkl')
        
        if gmm_labels3 is None:
            print("Computing GMM for lowres dataset")
            # Run GMM on UMAP embedding
            gmm_labels3 = compute_gmm(umap_embedding3)
            save_to_data_dir({'gmm_labels': gmm_labels3}, 'lowres_gmm_results.pkl')
    
    # Print summary of datasets and clusters
    print(f"\nDataset Summary:")
    print(f"Cryogen dataset: {len(gmm_labels1)} particles, {len(np.unique(gmm_labels1))} clusters")
    print(f"Original dataset: {len(gmm_labels2)} particles, {len(np.unique(gmm_labels2))} clusters")
    print(f"Lowres dataset: {len(gmm_labels3)} particles, {len(np.unique(gmm_labels3))} clusters")
    
    # Check if particles to label are in the dataset
    particle_indices = []
    for particle_id in PARTICLES_TO_LABEL:
        if particle_id < len(gmm_labels2):
            particle_indices.append(particle_id)
            print(f"Particle {particle_id} will be labeled, Cluster: {gmm_labels2[particle_id]}")
        else:
            print(f"Warning: Particle {particle_id} out of range ({len(gmm_labels2)} particles total)")
    
    # Print cluster correlations
    print_cluster_correlations(gmm_labels1, gmm_labels2)
    
    # Create hexbin plots
    print("\nCreating hexbin plots...")
    
    # 1. Dataset 1 UMAP with Dataset 2 GMM labels
    create_paper_hexbin(
        umap_embedding1,
        gmm_labels2,
        'UMAP from Cryogen Dataset with Clusters from Original Dataset',
        os.path.join(OUTPUT_DIR, 'umap1_gmm2_paper'),
        labeled_indices=particle_indices
    )
    
    # 2. Dataset 2 UMAP with its own GMM labels
    create_paper_hexbin(
        umap_embedding2,
        gmm_labels2,
        'UMAP from Original Dataset with its own Clusters',
        os.path.join(OUTPUT_DIR, 'umap2_gmm2_paper'),
        labeled_indices=particle_indices
    )
    
    # 3. Dataset 3 with gray hexbins
    create_paper_hexbin(
        umap_embedding3,
        gmm_labels3,
        'UMAP from Lowres Dataset',
        os.path.join(OUTPUT_DIR, 'umap3_gray_paper'),
        labeled_indices=particle_indices,
        use_gray_only=True,
        fixed_range=15.0 
    )
    
    print(f"\n Figures created in {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 