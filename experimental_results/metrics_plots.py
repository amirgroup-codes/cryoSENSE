import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Create output directory
output_dir = 'metrics_plots'
os.makedirs(output_dir, exist_ok=True)

# Set global font sizes
plt.rcParams['font.size'] = 16*1.5
plt.rcParams['axes.labelsize'] = 18*1.5
plt.rcParams['axes.titlesize'] = 18*1.5
plt.rcParams['xtick.labelsize'] = 14*1.5
plt.rcParams['ytick.labelsize'] = 14*1.5

# Load the data
df = pd.read_csv('reconstruction_metrics.csv')

# Filter data for different noise levels
df_zero_noise = df[df['noise_level'] == 0.0]
df_point1_noise = df[df['noise_level'] == 0.1]

# Define the protein names for each dataset
protein_names = {
    'empiar10166': 'EMPIAR10166_128',
    'empiar11526': 'EMPIAR11526_128',
    'empiar10786': 'EMPIAR10786_128',
    'empiar10076': 'empiar10076_128'
}

# Define display names for proteins
protein_display_names = {
    'EMPIAR10166_128': 'EMPIAR-10166',
    'EMPIAR11526_128': 'EMPIAR-11526',
    'EMPIAR10786_128': 'EMPIAR-10786',
    'empiar10076_128': 'EMPIAR-10076'
}

# Define new method names and colors
method_names = {
    'frugalcryo': 'CryoGEN',
    'tv_minimize': 'TV',
    'dct': 'DCT',
    'wavelet': 'Wavelet'
}

method_colors = {
    'frugalcryo': '#E54B35',  # red
    'tv_minimize': '#22628A',  # blue
    'dct': '#AB67BD',          # purple
    'wavelet': '#029448'       # green
}

# Define markers for each method
method_markers = {
    'frugalcryo': 'o',
    'tv_minimize': 'o',
    'dct': 'o',
    'wavelet': 'o'
}

# Define methods and downsampling factors
methods = ['frugalcryo', 'tv_minimize', 'dct', 'wavelet']
downsampling_factors = [2, 4, 8, 16, 32]

# Define a function to calculate mask ratio from num_masks and block_size
def get_mask_ratio(num_masks, block_size):
    total_masks = block_size * block_size
    return num_masks / total_masks

# PLOT 1: EMPIAR-10166 + EMPIAR-10076 LPIPS and SSIM with noise 0
def create_empiar10166_10076_lpips_ssim_plot():
    """
    Creates plot showing LPIPS and SSIM for EMPIAR-10166 and EMPIAR-10076
    with noise level 0.
    """
    # Define the proteins we want to show
    protein1 = 'EMPIAR10166_128'
    protein2 = 'empiar10076_128'
    
    # Create the plot structure (4 rows, 5 columns)
    # Row 1: LPIPS for protein1
    # Row 2: LPIPS for protein2
    # Row 3: SSIM for protein1
    # Row 4: SSIM for protein2
    metrics = ['LPIPS', 'LPIPS', 'SSIM', 'SSIM']
    proteins = [protein1, protein2, protein1, protein2]
    
    # Get y-limits for each metric to ensure consistency per row
    lpips_data = []
    ssim_data = []
    
    for protein in [protein1, protein2]:
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        if not protein_df.empty:
            lpips_data.extend(protein_df['LPIPS'].tolist())
            ssim_data.extend(protein_df['SSIM'].tolist())
    
    # Calculate min/max with padding
    lpips_min = min(lpips_data) - 0.05 * (max(lpips_data) - min(lpips_data))
    lpips_max = max(lpips_data) + 0.05 * (max(lpips_data) - min(lpips_data))
    ssim_min = min(ssim_data) - 0.05 * (max(ssim_data) - min(ssim_data))
    ssim_max = max(ssim_data) + 0.05 * (max(ssim_data) - min(ssim_data))
    
    # Set row-specific y-limits
    row_ylims = [
        [lpips_min, lpips_max],  # Row 1: LPIPS for protein1
        [lpips_min, lpips_max],  # Row 2: LPIPS for protein2
        [ssim_min, ssim_max],    # Row 3: SSIM for protein1
        [ssim_min, ssim_max]     # Row 4: SSIM for protein2
    ]
    
    # Create the figure
    fig, axs = plt.subplots(4, 5, figsize=(25, 14))
    plt.subplots_adjust(hspace=0.04, wspace=0.1)
    
    # Loop through the rows (metrics and proteins)
    for i in range(4):
        metric = metrics[i]
        protein = proteins[i]
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        
        # Loop through each downsampling factor (columns)
        for j, block_size in enumerate(downsampling_factors):
            ax = axs[i, j]
            block_df = protein_df[protein_df['block_size'] == block_size]
            
            # If no data for this block size, add a message
            if block_df.empty:
                ax.text(0.5, 0.5, f'No data for block size {block_size}', 
                       horizontalalignment='center', verticalalignment='center')
                continue
            
            # Plot each method
            for method in methods:
                if method not in method_colors:
                    continue
                    
                method_df = block_df[block_df['experiment'] == method]
                
                if method_df.empty:
                    continue
                
                # Group by num_masks to get mean and std
                grouped = method_df.groupby('num_masks')[metric].agg(['mean', 'std']).reset_index()
                
                # Handle cases where std might be NaN (only 1 sample)
                grouped['std'] = grouped['std'].fillna(0)
                
                # Convert num_masks to sampling ratio
                grouped['sampling_ratio'] = grouped['num_masks'].apply(
                    lambda x: get_mask_ratio(x, block_size))
                
                # Sort by sampling_ratio for proper line drawing
                grouped = grouped.sort_values('sampling_ratio')
                
                # Plot the line with markers
                line = ax.plot(grouped['sampling_ratio'], grouped['mean'], 
                              linestyle='-', 
                              marker=method_markers[method],
                              markersize=8,
                              label=method_names[method],
                              color=method_colors[method],
                              linewidth=2.5)
                
                # Add shaded error region
                ax.fill_between(grouped['sampling_ratio'], 
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=method_colors[method],
                               alpha=0.2)
            
            # Set y-limits based on the row
            ax.set_ylim(row_ylims[i])
            
            # Add explicit yticks for SSIM rows (but don't limit the y-axis)
            if metric == 'SSIM':
                ax.set_yticks([0.0, 0.4, 0.8])
            
            # Set x-axis limits and ticks
            if block_size == 2:
                ax.set_xlim([0.2, 1.05])  # Start at 0.2 for 2x
            else:
                ax.set_xlim([0, 1.05])    # Start at 0 for others
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'K={block_size}', fontsize=18*1.5, pad=10)
            
            if i == 3:  # Last row
                ax.set_xlabel(r'$\rho$', fontsize=18*1.5)
            
            # Add metric and protein info to the left side
            if j == 0:
                protein_display = protein_display_names[protein]
                ax.set_ylabel(f'{metric}', fontsize=18*1.5, labelpad=10)
                ax.text(-0.35, 0.5, f'{protein_display}', transform=ax.transAxes, 
                       size=18*1.5, ha='right', va='center', rotation=90)
            
            # Remove grid
            ax.grid(False)
            
            # Add legend to specific plots
            if i == 0 and j == 0:
                # Create legend for methods
                custom_lines = []
                custom_labels = []
                
                for method in methods:
                    if method in method_colors:
                        custom_lines.append(plt.Line2D([0], [0], 
                                                     color=method_colors[method], 
                                                     linestyle='-', 
                                                     marker='o',
                                                     markersize=8,
                                                     linewidth=2.5))
                        custom_labels.append(method_names[method])
                
                # Position legend in the upper right with smaller font
                legend = ax.legend(custom_lines, custom_labels,
                                  loc='upper right',
                                  fontsize=12, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    pdf_path = f'{output_dir}/EMPIAR-10166-10076_LPIPS_SSIM_noise0.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Created EMPIAR-10166 and EMPIAR-10076, LPIPS and SSIM plot: {pdf_path}")

# PLOT 2: EMPIAR-10166 + EMPIAR-10076 PSNR with noise 0
def create_empiar10166_10076_psnr_plot():
    """
    Creates plot showing PSNR for EMPIAR-10166 and EMPIAR-10076
    with noise level 0.
    """
    # Define the proteins we want to show
    proteins = ['EMPIAR10166_128', 'empiar10076_128']
    
    # Get y-limits for PSNR to ensure consistency
    psnr_data = []
    for protein in proteins:
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        if not protein_df.empty:
            psnr_data.extend(protein_df['PSNR'].tolist())
    
    # Calculate min/max with padding
    psnr_min = min(psnr_data) - 0.05 * (max(psnr_data) - min(psnr_data))
    psnr_max = max(psnr_data) + 0.05 * (max(psnr_data) - min(psnr_data))
    
    # Create the figure (2 rows, 5 columns)
    fig, axs = plt.subplots(2, 5, figsize=(25, 8))
    plt.subplots_adjust(hspace=0.04, wspace=0.1)
    
    # Loop through each protein (rows)
    for i, protein in enumerate(proteins):
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        
        # Loop through each downsampling factor (columns)
        for j, block_size in enumerate(downsampling_factors):
            ax = axs[i, j]
            block_df = protein_df[protein_df['block_size'] == block_size]
            
            # If no data for this block size, add a message
            if block_df.empty:
                ax.text(0.5, 0.5, f'No data for block size {block_size}', 
                      horizontalalignment='center', verticalalignment='center')
                continue
            
            # Plot each method
            for method in methods:
                if method not in method_colors:
                    continue
                    
                method_df = block_df[block_df['experiment'] == method]
                
                if method_df.empty:
                    continue
                
                # Group by num_masks to get mean and std for PSNR
                grouped = method_df.groupby('num_masks')['PSNR'].agg(['mean', 'std']).reset_index()
                
                # Handle cases where std might be NaN (only 1 sample)
                grouped['std'] = grouped['std'].fillna(0)
                
                # Convert num_masks to sampling ratio
                grouped['sampling_ratio'] = grouped['num_masks'].apply(
                    lambda x: get_mask_ratio(x, block_size))
                
                # Sort by sampling_ratio for proper line drawing
                grouped = grouped.sort_values('sampling_ratio')
                
                # Plot the line with markers
                line = ax.plot(grouped['sampling_ratio'], grouped['mean'], 
                              linestyle='-', 
                              marker='o',
                              markersize=8,
                              label=method_names[method],
                              color=method_colors[method],
                              linewidth=2.5)
                
                # Add shaded error region
                ax.fill_between(grouped['sampling_ratio'], 
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=method_colors[method],
                               alpha=0.2)
            
            # Set consistent y-limits for PSNR
            ax.set_ylim([psnr_min, psnr_max])
            
            # Set x-axis limits and ticks
            if block_size == 2:
                ax.set_xlim([0.2, 1.05])  # Start at 0.2 for 2x
            else:
                ax.set_xlim([0, 1.05])    # Start at 0 for others
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'K={block_size}', fontsize=18*1.5, pad=10)
            
            if i == 1:  # Last row
                ax.set_xlabel(r'$\rho$', fontsize=18*1.5)
            
            # Add protein info to the left side
            if j == 0:
                protein_display = protein_display_names[protein]
                ax.set_ylabel('PSNR (dB)', fontsize=18*1.5, labelpad=10)
                ax.text(-0.35, 0.5, f'{protein_display}', transform=ax.transAxes, 
                       size=18*1.5, ha='right', va='center', rotation=90)
            
            # Remove grid
            ax.grid(False)
            
            # Add legend to first plot
            if i == 0 and j == 0:
                # Create legend for methods
                custom_lines = []
                custom_labels = []
                
                for method in methods:
                    if method in method_colors:
                        custom_lines.append(plt.Line2D([0], [0], 
                                                     color=method_colors[method], 
                                                     linestyle='-', 
                                                     marker='o',
                                                     markersize=8,
                                                     linewidth=2.5))
                        custom_labels.append(method_names[method])
                
                # Position legend in the lower right with smaller font
                legend = ax.legend(custom_lines, custom_labels,
                                  loc='lower right',
                                  fontsize=12, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    pdf_path = f'{output_dir}/EMPIAR-10166-10076_PSNR_noise0.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Created EMPIAR-10166 and EMPIAR-10076, PSNR plot: {pdf_path}")

# PLOT 3: EMPIAR-10786 + EMPIAR-11526 LPIPS and SSIM with noise 0
def create_empiar10786_11526_lpips_ssim_plot():
    """
    Creates plot showing LPIPS and SSIM for EMPIAR-10786 and EMPIAR-11526
    with noise level 0.
    """
    # Define the proteins we want to show
    protein1 = 'EMPIAR10786_128'
    protein2 = 'EMPIAR11526_128'
    
    # Create the plot structure (4 rows, 5 columns)
    # Row 1: LPIPS for protein1
    # Row 2: LPIPS for protein2
    # Row 3: SSIM for protein1
    # Row 4: SSIM for protein2
    metrics = ['LPIPS', 'LPIPS', 'SSIM', 'SSIM']
    proteins = [protein1, protein2, protein1, protein2]
    
    # Get y-limits for each metric to ensure consistency per row
    lpips_data = []
    ssim_data = []
    
    for protein in [protein1, protein2]:
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        if not protein_df.empty:
            lpips_data.extend(protein_df['LPIPS'].tolist())
            ssim_data.extend(protein_df['SSIM'].tolist())
    
    # Calculate min/max with padding
    lpips_min = min(lpips_data) - 0.05 * (max(lpips_data) - min(lpips_data))
    lpips_max = max(lpips_data) + 0.05 * (max(lpips_data) - min(lpips_data))
    ssim_min = min(ssim_data) - 0.05 * (max(ssim_data) - min(ssim_data))
    ssim_max = max(ssim_data) + 0.05 * (max(ssim_data) - min(ssim_data))
    
    # Set row-specific y-limits
    row_ylims = [
        [lpips_min, lpips_max],  # Row 1: LPIPS for protein1
        [lpips_min, lpips_max],  # Row 2: LPIPS for protein2
        [ssim_min, ssim_max],    # Row 3: SSIM for protein1
        [ssim_min, ssim_max]     # Row 4: SSIM for protein2
    ]
    
    # Create the figure
    fig, axs = plt.subplots(4, 5, figsize=(25, 14))
    plt.subplots_adjust(hspace=0.04, wspace=0.1)
    
    # Loop through the rows (metrics and proteins)
    for i in range(4):
        metric = metrics[i]
        protein = proteins[i]
        protein_df = df_zero_noise[df_zero_noise['protein'] == protein]
        
        # Loop through each downsampling factor (columns)
        for j, block_size in enumerate(downsampling_factors):
            ax = axs[i, j]
            block_df = protein_df[protein_df['block_size'] == block_size]
            
            # If no data for this block size, add a message
            if block_df.empty:
                ax.text(0.5, 0.5, f'No data for block size {block_size}', 
                       horizontalalignment='center', verticalalignment='center')
                continue
            
            # Plot each method
            for method in methods:
                if method not in method_colors:
                    continue
                    
                method_df = block_df[block_df['experiment'] == method]
                
                if method_df.empty:
                    continue
                
                # Group by num_masks to get mean and std
                grouped = method_df.groupby('num_masks')[metric].agg(['mean', 'std']).reset_index()
                
                # Handle cases where std might be NaN (only 1 sample)
                grouped['std'] = grouped['std'].fillna(0)
                
                # Convert num_masks to sampling ratio
                grouped['sampling_ratio'] = grouped['num_masks'].apply(
                    lambda x: get_mask_ratio(x, block_size))
                
                # Sort by sampling_ratio for proper line drawing
                grouped = grouped.sort_values('sampling_ratio')
                
                # Plot the line with markers
                line = ax.plot(grouped['sampling_ratio'], grouped['mean'], 
                              linestyle='-', 
                              marker=method_markers[method],
                              markersize=8,
                              label=method_names[method],
                              color=method_colors[method],
                              linewidth=2.5)
                
                # Add shaded error region
                ax.fill_between(grouped['sampling_ratio'], 
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=method_colors[method],
                               alpha=0.2)
            
            # Set y-limits based on the row
            ax.set_ylim(row_ylims[i])
            
            # Add explicit yticks for SSIM rows (but don't limit the y-axis)
            if metric == 'SSIM':
                ax.set_yticks([0.0, 0.4, 0.8])
            
            # Set x-axis limits and ticks
            if block_size == 2:
                ax.set_xlim([0.2, 1.05])  # Start at 0.2 for 2x
            else:
                ax.set_xlim([0, 1.05])    # Start at 0 for others
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'K={block_size}', fontsize=18*1.5, pad=10)
            
            if i == 3:  # Last row
                ax.set_xlabel(r'$\rho$', fontsize=18*1.5)
            
            # Add metric and protein info to the left side
            if j == 0:
                protein_display = protein_display_names[protein]
                ax.set_ylabel(f'{metric}', fontsize=18*1.5, labelpad=10)
                ax.text(-0.35, 0.5, f'{protein_display}', transform=ax.transAxes, 
                       size=18*1.5, ha='right', va='center', rotation=90)
            
            # Remove grid
            ax.grid(False)
            
            # Add legend to specific plots - moving to first subplot as requested
            if i == 0 and j == 0:
                # Create legend for methods
                custom_lines = []
                custom_labels = []
                
                for method in methods:
                    if method in method_colors:
                        custom_lines.append(plt.Line2D([0], [0], 
                                                     color=method_colors[method], 
                                                     linestyle='-', 
                                                     marker='o',
                                                     markersize=8,
                                                     linewidth=2.5))
                        custom_labels.append(method_names[method])
                
                # Position legend in the upper right with smaller font
                legend = ax.legend(custom_lines, custom_labels,
                                  loc='upper right',
                                  fontsize=12, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    pdf_path = f'{output_dir}/EMPIAR-10786-11526_LPIPS_SSIM_noise0.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Created EMPIAR-10786 and EMPIAR-11526, LPIPS and SSIM plot: {pdf_path}")

# PLOT 4: All proteins LPIPS and SSIM with noise 0.1
def create_all_proteins_lpips_ssim_noise01_plot():
    """
    Creates plot showing LPIPS and SSIM for all proteins with noise level 0.1.
    """
    # Define the proteins we want to include
    all_proteins = ['EMPIAR10166_128', 'EMPIAR11526_128', 'EMPIAR10786_128', 'empiar10076_128']
    
    # Number of proteins and metrics
    num_proteins = len(all_proteins)
    num_metrics = 2  # LPIPS and SSIM
    num_rows = num_proteins * num_metrics
    
    # Organize metrics and proteins for each row
    metrics = []
    proteins = []
    
    # First all LPIPS rows, then all SSIM rows
    for metric in ['LPIPS', 'SSIM']:
        for protein in all_proteins:
            metrics.append(metric)
            proteins.append(protein)
    
    # Get y-limits for each metric to ensure consistency per metric
    lpips_data = []
    ssim_data = []
    
    for protein in all_proteins:
        protein_df = df_point1_noise[df_point1_noise['protein'] == protein]
        if not protein_df.empty:
            lpips_data.extend(protein_df['LPIPS'].tolist())
            ssim_data.extend(protein_df['SSIM'].tolist())
    
    # Calculate min/max with padding
    lpips_min = min(lpips_data) - 0.05 * (max(lpips_data) - min(lpips_data))
    lpips_max = max(lpips_data) + 0.05 * (max(lpips_data) - min(lpips_data))
    ssim_min = min(ssim_data) - 0.05 * (max(ssim_data) - min(ssim_data))
    ssim_max = max(ssim_data) + 0.05 * (max(ssim_data) - min(ssim_data))
    
    # Create the figure
    fig, axs = plt.subplots(num_rows, 5, figsize=(25, 4 * num_rows))
    plt.subplots_adjust(hspace=0.04, wspace=0.1)
    
    # Loop through the rows (metrics and proteins)
    for i in range(num_rows):
        metric = metrics[i]
        protein = proteins[i]
        protein_df = df_point1_noise[df_point1_noise['protein'] == protein]
        
        # Loop through each downsampling factor (columns)
        for j, block_size in enumerate(downsampling_factors):
            ax = axs[i, j]
            block_df = protein_df[protein_df['block_size'] == block_size]
            
            # If no data for this block size, add a message
            if block_df.empty:
                ax.text(0.5, 0.5, f'No data for block size {block_size}', 
                       horizontalalignment='center', verticalalignment='center')
                continue
            
            # Plot each method
            for method in methods:
                if method not in method_colors:
                    continue
                    
                method_df = block_df[block_df['experiment'] == method]
                
                if method_df.empty:
                    continue
                
                # Group by num_masks to get mean and std
                grouped = method_df.groupby('num_masks')[metric].agg(['mean', 'std']).reset_index()
                
                # Handle cases where std might be NaN (only 1 sample)
                grouped['std'] = grouped['std'].fillna(0)
                
                # Convert num_masks to sampling ratio
                grouped['sampling_ratio'] = grouped['num_masks'].apply(
                    lambda x: get_mask_ratio(x, block_size))
                
                # Sort by sampling_ratio for proper line drawing
                grouped = grouped.sort_values('sampling_ratio')
                
                # Plot the line with markers
                line = ax.plot(grouped['sampling_ratio'], grouped['mean'], 
                              linestyle='-', 
                              marker=method_markers[method],
                              markersize=8,
                              label=method_names[method],
                              color=method_colors[method],
                              linewidth=2.5)
                
                # Add shaded error region
                ax.fill_between(grouped['sampling_ratio'], 
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=method_colors[method],
                               alpha=0.2)
            
            # Set y-limits based on the metric
            if metric == 'LPIPS':
                ax.set_ylim([lpips_min, lpips_max])
            elif metric == 'SSIM':
                ax.set_ylim([ssim_min, ssim_max])
                ax.set_yticks([0.0, 0.4, 0.8])
            
            # Set x-axis limits and ticks
            if block_size == 2:
                ax.set_xlim([0.2, 1.05])  # Start at 0.2 for 2x
            else:
                ax.set_xlim([0, 1.05])    # Start at 0 for others
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set titles and labels
            if i == 0:
                ax.set_title(f'K={block_size}', fontsize=18*1.5, pad=10)
            
            if i == num_rows - 1:  # Last row
                ax.set_xlabel(r'$\rho$', fontsize=18*1.5)
            
            # Add metric and protein info to the left side
            if j == 0:
                protein_display = protein_display_names[protein]
                ax.set_ylabel(f'{metric}', fontsize=18*1.5, labelpad=10)
                ax.text(-0.35, 0.5, f'{protein_display}', transform=ax.transAxes, 
                       size=18*1.5, ha='right', va='center', rotation=90)
            
            # Remove grid
            ax.grid(False)
            
            # Add legend to the first subplot (first row, first column)
            if i == 0 and j == 0:
                # Create legend for methods
                custom_lines = []
                custom_labels = []
                
                for method in methods:
                    if method in method_colors:
                        custom_lines.append(plt.Line2D([0], [0], 
                                                     color=method_colors[method], 
                                                     linestyle='-', 
                                                     marker='o',
                                                     markersize=8,
                                                     linewidth=2.5))
                        custom_labels.append(method_names[method])
                
                # Position legend in the upper right with smaller font
                legend = ax.legend(custom_lines, custom_labels,
                                  loc='upper right',
                                  fontsize=12, ncol=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    pdf_path = f'{output_dir}/all_proteins_LPIPS_SSIM_noise01.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Created all proteins, LPIPS and SSIM with noise 0.1 plot: {pdf_path}")

# Create all plots
create_empiar10166_10076_lpips_ssim_plot()
create_empiar10166_10076_psnr_plot()
create_empiar10786_11526_lpips_ssim_plot()
create_all_proteins_lpips_ssim_noise01_plot()

print("All plots created successfully!") 