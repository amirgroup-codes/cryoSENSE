import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import MMCIFParser, Superimposer
import os
from Bio import pairwise2
from Bio.SeqUtils import seq1
import numpy as np
import math
from collections import Counter

# Minimum chain length to consider for any analysis
MIN_CHAIN_LENGTH = 20

def extract_b_factors(cif_file):
    """Extract B-factors from the model file"""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_file)
    b_factors = [atom.bfactor for atom in structure.get_atoms() if atom.element != 'H']
    return b_factors

def get_chain_sequences(cif_file):
    """Extract amino acid sequences from all chains in the model."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_file)
    
    chain_sequences = {}
    total_chains = 0
    filtered_chains = 0
    
    for model in structure:
        for chain in model:
            total_chains += 1
            chain_id = chain.id
            # Convert residues to single letter amino acid code
            sequence = ""
            for residue in chain:
                if residue.get_resname() in ["HOH", "WAT"]:  # Skip water molecules
                    continue
                try:
                    aa = seq1(residue.get_resname())
                    sequence += aa if aa != "X" else ""
                except:
                    continue  # Skip non-standard residues
            
            # Only include chains with MIN_CHAIN_LENGTH or more residues
            if sequence and len(sequence) >= MIN_CHAIN_LENGTH:
                chain_sequences[chain_id] = sequence
            else:
                filtered_chains += 1
    
    # print(f"  Total chains: {total_chains}, Filtered out {filtered_chains} chains with <{MIN_CHAIN_LENGTH} residues")
    return chain_sequences

def align_chains(seq1, seq2, min_length=10, min_identity=0.7):
    """Perform pairwise alignment between two sequences."""
    if len(seq1) < min_length or len(seq2) < min_length:
        return None
    
    # Perform global alignment with custom scoring
    alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -2, -0.5, one_alignment_only=True)
    
    if not alignments:
        return None
    
    alignment = alignments[0]
    aligned_seq1, aligned_seq2, score, begin, end = alignment
    
    # Calculate identity
    matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' and b != '-')
    alignment_length = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != '-' or b != '-')
    
    if alignment_length == 0:
        return None
    
    identity = matches / alignment_length
    
    if identity >= min_identity:
        return {
            "score": score,
            "identity": identity,
            "alignment_length": alignment_length,
            "seq1_length": len(seq1),
            "seq2_length": len(seq2),
            "aligned_seq1": aligned_seq1,
            "aligned_seq2": aligned_seq2
        }
    
    return None

def calculate_backbone_rmsd(model1_path, chain1_id, model2_path, chain2_id, aligned_seq1, aligned_seq2):
    """Calculate backbone RMSD between aligned portions of two chains using all backbone atoms (N, CA, C, O)."""
    # Parse structures
    parser = MMCIFParser(QUIET=True)
    structure1 = parser.get_structure("model1", model1_path)
    structure2 = parser.get_structure("model2", model2_path)
    
    # Get chains
    chain1 = structure1[0][chain1_id]
    chain2 = structure2[0][chain2_id]
    
    # Define backbone atoms to include
    backbone_atoms = ["N", "CA", "C", "O"]
    
    # Get backbone atoms from aligned regions (excluding gaps)
    atoms1 = []
    atoms2 = []
    
    # Track residue indices for both chains
    idx1 = 0
    idx2 = 0
    
    # Iterate through alignment
    for a1, a2 in zip(aligned_seq1, aligned_seq2):
        if a1 != '-':
            res1_idx = idx1
            idx1 += 1
        if a2 != '-':
            res2_idx = idx2
            idx2 += 1
            
        # Only consider positions where both sequences have residues (no gaps)
        if a1 != '-' and a2 != '-':
            try:
                # Get residues by index (skip non-standard residues)
                res1 = list(chain1.get_residues())[res1_idx]
                res2 = list(chain2.get_residues())[res2_idx]
                
                # Check for each backbone atom and add if available
                for atom_name in backbone_atoms:
                    if atom_name in res1 and atom_name in res2:
                        atoms1.append(res1[atom_name])
                        atoms2.append(res2[atom_name])
            except (IndexError, KeyError):
                # Skip if residue index is out of range or no atom
                continue
    
    # Calculate RMSD if we have enough atoms
    if len(atoms1) >= 3 and len(atoms2) >= 3:
        super_imposer = Superimposer()
        super_imposer.set_atoms(atoms1, atoms2)
        return super_imposer.rms, len(atoms1)
    
    return None, 0

def calculate_spatial_distance(model1_path, chain1_id, model2_path, chain2_id, aligned_seq1, aligned_seq2):
    """Calculate spatial distance between aligned portions of two chains WITHOUT superimposition."""
    # Parse structures
    parser = MMCIFParser(QUIET=True)
    structure1 = parser.get_structure("model1", model1_path)
    structure2 = parser.get_structure("model2", model2_path)
    
    # Get chains
    chain1 = structure1[0][chain1_id]
    chain2 = structure2[0][chain2_id]
    
    # Track residue indices for both chains
    idx1 = 0
    idx2 = 0
    
    # Lists to store distances
    distances = []
    
    # Get centroids for each chain
    centroid1_coords = np.zeros(3)
    centroid2_coords = np.zeros(3)
    n_atoms1 = 0
    n_atoms2 = 0
    
    # First pass to calculate centroids
    for a1, a2 in zip(aligned_seq1, aligned_seq2):
        if a1 != '-':
            res1_idx = idx1
            idx1 += 1
        if a2 != '-':
            res2_idx = idx2
            idx2 += 1
            
        # Only consider positions where both sequences have residues (no gaps)
        if a1 != '-' and a2 != '-':
            try:
                # Get residues by index
                res1 = list(chain1.get_residues())[res1_idx]
                res2 = list(chain2.get_residues())[res2_idx]
                
                # Check if both residues have CA atoms
                if 'CA' in res1:
                    centroid1_coords += res1['CA'].coord
                    n_atoms1 += 1
                if 'CA' in res2:
                    centroid2_coords += res2['CA'].coord
                    n_atoms2 += 1
            except (IndexError, KeyError):
                continue
    
    if n_atoms1 == 0 or n_atoms2 == 0:
        return None, 0
    
    centroid1_coords /= n_atoms1
    centroid2_coords /= n_atoms2
    
    # Calculate centroid-to-centroid distance
    centroid_distance = math.sqrt(np.sum((centroid1_coords - centroid2_coords)**2))
    
    # Reset indices for second pass
    idx1 = 0
    idx2 = 0
    
    # Second pass to calculate distances between corresponding atoms
    for a1, a2 in zip(aligned_seq1, aligned_seq2):
        if a1 != '-':
            res1_idx = idx1
            idx1 += 1
        if a2 != '-':
            res2_idx = idx2
            idx2 += 1
            
        # Only consider positions where both sequences have residues (no gaps)
        if a1 != '-' and a2 != '-':
            try:
                # Get residues by index
                res1 = list(chain1.get_residues())[res1_idx]
                res2 = list(chain2.get_residues())[res2_idx]
                
                # Check if both residues have CA atoms
                if 'CA' in res1 and 'CA' in res2:
                    # Calculate Euclidean distance between CA atoms
                    dist = math.sqrt(sum((res1['CA'].coord[i] - res2['CA'].coord[i])**2 for i in range(3)))
                    distances.append(dist)
            except (IndexError, KeyError):
                continue
    
    # Calculate average distance if we have enough atoms
    if len(distances) >= 3:
        avg_distance = sum(distances) / len(distances)
        return {
            "centroid_distance": centroid_distance,
            "avg_atom_distance": avg_distance,
            "min_atom_distance": min(distances),
            "max_atom_distance": max(distances),
            "atoms_compared": len(distances)
        }
    
    return None, 0

def find_similar_chains_across_models(model_sequences, model_paths, min_length=20, min_identity=0.7):
    """Find similar chains across different models based on sequence alignment."""
    model_names = list(model_sequences.keys())
    similar_chains = []
    model_pair_stats = {}
    
    # Count total number of pairs to analyze
    total_pairs = 0
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            chains1_count = len(model_sequences[model1])
            chains2_count = len(model_sequences[model2])
            total_pairs += chains1_count * chains2_count
            
            # Initialize stats for this model pair
            model_pair_stats[(model1, model2)] = {
                "total_comparisons": chains1_count * chains2_count,
                "alignments_found": 0,
                "rmsd_calculated": 0,
                "backbone_rmsd_calculated": 0,
                "spatial_distance_calculated": 0,
                "matches_within_3A": 0
            }
    
    print(f"\nAnalyzing a total of {total_pairs} chain pairs...")
    current_pair = 0
    
    # Compare each pair of models
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            chains1 = model_sequences[model1]
            chains2 = model_sequences[model2]
            
            print(f"\nComparing {model1} ({len(chains1)} chains) with {model2} ({len(chains2)} chains)")
            model_pair_count = 0
            
            for chain1_id, seq1 in chains1.items():
                for chain2_id, seq2 in chains2.items():
                    current_pair += 1
                    model_pair_count += 1
                    
                    # Print progress every 50 pairs or at the end of each model pair
                    if current_pair % 50 == 0 or model_pair_count == len(chains1) * len(chains2):
                        progress_pct = (current_pair / total_pairs) * 100
                        print(f"Progress: {current_pair}/{total_pairs} pairs analyzed ({progress_pct:.1f}%)", end="\r")
                    
                    alignment = align_chains(seq1, seq2, min_length, min_identity)
                    if alignment:
                        model_pair_stats[(model1, model2)]["alignments_found"] += 1
                        
                        # Calculate backbone RMSD (N, CA, C, O atoms)
                        backbone_rmsd, backbone_aligned_atoms = calculate_backbone_rmsd(
                            model_paths[model1], chain1_id,
                            model_paths[model2], chain2_id,
                            alignment["aligned_seq1"], alignment["aligned_seq2"]
                        )
                        
                        if backbone_rmsd is not None:
                            model_pair_stats[(model1, model2)]["backbone_rmsd_calculated"] += 1
                        
                        # Calculate spatial distance WITHOUT superimposition
                        spatial_distance = calculate_spatial_distance(
                            model_paths[model1], chain1_id, 
                            model_paths[model2], chain2_id,
                            alignment["aligned_seq1"], alignment["aligned_seq2"]
                        )
                        
                        if spatial_distance is not None and isinstance(spatial_distance, dict):
                            model_pair_stats[(model1, model2)]["spatial_distance_calculated"] += 1
                            
                            if spatial_distance["centroid_distance"] <= 3.0:
                                model_pair_stats[(model1, model2)]["matches_within_3A"] += 1
                        
                        similar_chains.append({
                            "model1": model1,
                            "chain1": chain1_id,
                            "model2": model2,
                            "chain2": chain2_id,
                            "alignment": alignment,
                            "backbone_rmsd": backbone_rmsd,
                            "backbone_aligned_atoms": backbone_aligned_atoms,
                            "spatial_distance": spatial_distance
                        })
            
            # print() # New line after model pair is complete
    
    # Print model pair statistics
    # print("\nStatistics by model pair:")
    # for (model1, model2), stats in model_pair_stats.items():
    #     print(f"\n{model1} vs {model2}:")
    #     print(f"  Total comparisons: {stats['total_comparisons']}")
    #     print(f"  Alignments found: {stats['alignments_found']}")
    #     print(f"  Backbone RMSD calculated: {stats['backbone_rmsd_calculated']}")
    #     print(f"  Spatial distance calculated: {stats['spatial_distance_calculated']}")
    #     print(f"  Matches within 3Å: {stats['matches_within_3A']}")
    
    # print(f"\nFound {len(similar_chains)} similar chain pairs out of {total_pairs} analyzed.")
    return similar_chains

def print_and_save_results(similar_chains, output_file):
    """Print and save the alignment results."""
    # Count matches by model pairs
    model_pairs = {}
    for match in similar_chains:
        pair = (match["model1"], match["model2"])
        if pair not in model_pairs:
            model_pairs[pair] = 0
        model_pairs[pair] += 1
    
    # Print results
    # print(f"\nSimilar chains found across models: {len(similar_chains)} matches")
    
    # print("\nMatches by model pair:")
    # for pair, count in model_pairs.items():
    #     print(f"  {pair[0]} vs {pair[1]}: {count} matches")
    
    # Save the fragment alignment data
    with open(output_file, "w") as f:
        f.write("Fragment Alignments Across Models\n")
        f.write("================================\n\n")
        
        f.write("Matches by model pair:\n")
        for pair, count in model_pairs.items():
            f.write(f"{pair[0]} vs {pair[1]}: {count} matches\n")
        f.write("\n")
        
        if similar_chains:
            for match in similar_chains:
                model1 = match["model1"]
                chain1 = match["chain1"]
                model2 = match["model2"]
                chain2 = match["chain2"]
                alignment = match["alignment"]
                backbone_rmsd = match.get("backbone_rmsd")
                backbone_aligned_atoms = match.get("backbone_aligned_atoms")
                spatial_distance = match["spatial_distance"]
                
                f.write(f"{model1} Chain {chain1} ({alignment['seq1_length']} residues) matches\n")
                f.write(f"{model2} Chain {chain2} ({alignment['seq2_length']} residues)\n")
                f.write(f"Sequence identity: {alignment['identity']:.2f}, Score: {alignment['score']:.1f}, Alignment length: {alignment['alignment_length']}\n")
                
                if backbone_rmsd is not None:
                    f.write(f"Backbone RMSD (N,CA,C,O) after superimposition: {backbone_rmsd:.2f} Å over {backbone_aligned_atoms} backbone atoms\n")
                else:
                    f.write("Backbone RMSD: Could not calculate (insufficient aligned atoms)\n")
                    
                if spatial_distance is not None and spatial_distance != 0:
                    f.write(f"Actual 3D position difference:\n")
                    f.write(f"  Centroid distance: {spatial_distance['centroid_distance']:.2f} Å\n")
                    f.write(f"  Average atom distance: {spatial_distance['avg_atom_distance']:.2f} Å\n")
                    f.write(f"  Min/Max atom distance: {spatial_distance['min_atom_distance']:.2f}/{spatial_distance['max_atom_distance']:.2f} Å\n")
                else:
                    f.write("Spatial distance: Could not calculate\n")
                    
                f.write(f"Alignment:\n")
                f.write(f"{alignment['aligned_seq1']}\n")
                f.write(f"{alignment['aligned_seq2']}\n\n")
        else:
            f.write("No similar chains found with the current threshold.")

def find_closest_chains_by_spatial_proximity(model1_path, target_chains, model2_path, max_centroid_distance=20.0):
    """
    Find chains in model2 that are spatially closest to specific chains in model1,
    regardless of sequence similarity.
    """
    # print(f"\nFinding closest chains in spatial proximity (max distance: {max_centroid_distance}Å)...")
    
    # Parse structures
    parser = MMCIFParser(QUIET=True)
    structure1 = parser.get_structure("model1", model1_path)
    structure2 = parser.get_structure("model2", model2_path)
    
    # Extract chains from both models
    model1_chains = {}
    for chain in structure1[0]:
        chain_id = chain.id
        model1_chains[chain_id] = chain
    
    model2_chains = {}
    for chain in structure2[0]:
        chain_id = chain.id
        model2_chains[chain_id] = chain
    
    # Filter to only target chains
    target_chain_objects = {chain_id: model1_chains[chain_id] for chain_id in target_chains if chain_id in model1_chains}
    
    if not target_chain_objects:
        print(f"Warning: None of the specified target chains {target_chains} were found in model1")
        return {}
    
    # Calculate centroids and atom distances for all chains
    def calculate_chain_metrics(chain):
        coords = np.zeros(3)
        count = 0
        for residue in chain:
            if 'CA' in residue:
                coords += residue['CA'].coord
                count += 1
        if count == 0:
            return None, None
        return coords / count, count
    
    # Get centroids for target chains
    target_metrics = {}
    for chain_id, chain in target_chain_objects.items():
        centroid, atom_count = calculate_chain_metrics(chain)
        if centroid is not None:
            target_metrics[chain_id] = (centroid, atom_count)
    
    # Get centroids for all model2 chains
    model2_metrics = {}
    for chain_id, chain in model2_chains.items():
        centroid, atom_count = calculate_chain_metrics(chain)
        if centroid is not None:
            model2_metrics[chain_id] = (centroid, atom_count)
    
    # Find closest chains for each target
    closest_chains = {}
    for target_id, (target_centroid, target_atoms) in target_metrics.items():
        # Calculate distance to each model2 chain
        distances = []
        for model2_id, (model2_centroid, model2_atoms) in model2_metrics.items():
            # Calculate centroid distance
            centroid_distance = math.sqrt(np.sum((target_centroid - model2_centroid)**2))
            
            # Skip if beyond max distance
            if centroid_distance > max_centroid_distance:
                continue
            
            # Calculate atom-to-atom distances
            atom_distances = []
            target_chain = target_chain_objects[target_id]
            model2_chain = model2_chains[model2_id]
            
            # Get all CA atoms from both chains
            target_ca_atoms = [res['CA'] for res in target_chain if 'CA' in res]
            model2_ca_atoms = [res['CA'] for res in model2_chain if 'CA' in res]
            
            # Calculate distances between all pairs of CA atoms
            for t_atom in target_ca_atoms:
                for m_atom in model2_ca_atoms:
                    dist = math.sqrt(sum((t_atom.coord[i] - m_atom.coord[i])**2 for i in range(3)))
                    atom_distances.append(dist)
            
            if atom_distances:
                avg_distance = sum(atom_distances) / len(atom_distances)
                min_distance = min(atom_distances)
                max_distance = max(atom_distances)
                
                distances.append({
                    'chain_id': model2_id,
                    'centroid_distance': centroid_distance,
                    'avg_atom_distance': avg_distance,
                    'min_atom_distance': min_distance,
                    'max_atom_distance': max_distance,
                    'atom_count': model2_atoms
                })
        
        # Sort by centroid distance
        distances.sort(key=lambda x: x['centroid_distance'])
        
        # Store all chains within max distance
        closest_chains[target_id] = distances
    
    return closest_chains

def save_proximity_results(closest_chains, model1_name, model2_name, output_file):
    """Save the proximity analysis results to a file"""
    with open(output_file, "w") as f:
        f.write(f"Spatially Closest Chains from {model2_name} to Target Chains in {model1_name}\n")
        f.write("==================================================================\n\n")
        
        for target_id, matches in closest_chains.items():
            f.write(f"{model1_name} Chain {target_id} matches in {model2_name}:\n")
            for i, dist_info in enumerate(matches, 1):
                f.write(f"  {i}. Chain {dist_info['chain_id']}:\n")
                f.write(f"     Centroid distance: {dist_info['centroid_distance']:.2f} Å\n")
                f.write(f"     Average atom distance: {dist_info['avg_atom_distance']:.2f} Å\n")
                f.write(f"     Min/Max atom distance: {dist_info['min_atom_distance']:.2f}/{dist_info['max_atom_distance']:.2f} Å\n")
                f.write(f"     Number of atoms: {dist_info['atom_count']}\n")
            f.write("\n")

def generate_confidence_violin_plot(b_factors_all, output_file):
    """Generate a violin plot of model confidence scores"""
    plt.figure(figsize=(6, 8))
    data = [b_factors_all[name] for name in b_factors_all.keys()]
    
    # Calculate mean and standard deviation
    means = []
    stds = []
    for name in b_factors_all.keys():
        mean_val = np.mean(b_factors_all[name])
        std_val = np.std(b_factors_all[name])
        means.append(mean_val)
        stds.append(std_val)
        # print(f"  {name}: mean = {mean_val:.2f} ± {std_val:.2f}")
    
    violin = plt.violinplot(dataset=data, 
                          showextrema=False,  # Remove the min/max lines
                          showmedians=False)  # Remove the median line
    
    # Set the color to skyblue
    for pc in violin['bodies']:
        pc.set_facecolor('#87CEEB')
        pc.set_alpha(0.7)
        pc.set_edgecolor('none')  # Remove edges
    
    plt.xticks(ticks=range(1, len(b_factors_all) + 1), labels=list(b_factors_all.keys()))
    plt.ylabel("Confidence Scores")
    plt.title("Confidence Score Distribution")
    plt.grid(False)
    
    # Add mean values as text
    for i, (mean_val, std_val) in enumerate(zip(means, stds)):
        plt.text(i+1, np.max(data[i]), f"μ = {mean_val:.2f}\nσ = {std_val:.2f}", ha='center', va='top')
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def generate_alignment_vs_identity_scatter(similar_chains, output_file):
    """Generate scatter plot of alignment score vs sequence identity"""
    plt.figure(figsize=(10, 8))
    
    # Define custom colors
    custom_colors = {
        "original vs cryogen": "#96D7E6",       # Light blue for cryogen points
        "original vs lowres": "#FFA197"         # Light salmon for lowres points
    }
    
    # Filter to keep only original-cryogen and original-lowres pairs
    filtered_chains = []
    comparison_groups = {
        "original vs cryogen": [],
        "original vs lowres": []
    }
    
    for match in similar_chains:
        model1 = match["model1"]
        model2 = match["model2"]
        
        # Keep only pairs involving original
        if (model1 == "original" and (model2 == "cryogen" or model2 == "lowres")) or \
           (model2 == "original" and (model1 == "cryogen" or model1 == "lowres")):
            filtered_chains.append(match)
            
            # Group by comparison type
            if (model1 == "original" and model2 == "cryogen") or (model2 == "original" and model1 == "cryogen"):
                comparison_groups["original vs cryogen"].append(match)
            elif (model1 == "original" and model2 == "lowres") or (model2 == "original" and model1 == "lowres"):
                comparison_groups["original vs lowres"].append(match)
    
    # Generate the scatter plot
    for comparison, matches in comparison_groups.items():
        x = []  # Sequence identity (%)
        y = []  # Alignment score
        rmsd_values = []  # Backbone RMSD for marker size
        
        for match in matches:
            if match.get("backbone_rmsd") is not None:
                x.append(match["alignment"]["identity"] * 100)  # Convert to percentage
                y.append(match["alignment"]["score"])
                rmsd_values.append(match["backbone_rmsd"])
        
        # For marker size, use inverse of RMSD (small RMSD = large marker)
        marker_sizes = [1000 / (rmsd + 1) for rmsd in rmsd_values]  # Add 1 to avoid division by zero
        
        plt.scatter(x, y, alpha=0.7, c=custom_colors[comparison], 
                   label=comparison, s=marker_sizes)
    
    plt.xlabel("Sequence Identity (%)", fontsize=14)
    plt.ylabel("Alignment Score", fontsize=14)
    plt.title("Alignment Score vs Sequence Identity\n(Marker size inversely proportional to Backbone RMSD)", fontsize=16)
    plt.grid(False)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Create two legends - one for comparison groups and one for sizes
    handles, labels = plt.gca().get_legend_handles_labels()
    first_legend = plt.legend(handles=handles, labels=labels, loc="upper right", fontsize=12)
    plt.gca().add_artist(first_legend)
    
    # Add a size legend
    rmsd_examples = [1.0, 2.0, 4.0, 8.0]  # Example RMSD values for legend
    size_handles = []
    size_labels = []
    for rmsd in rmsd_examples:
        size = 1000 / (rmsd + 1)
        handle = plt.scatter([], [], alpha=0.7, c='gray', s=size)
        size_handles.append(handle)
        size_labels.append(f"RMSD {rmsd:.1f}Å")
    second_legend = plt.legend(handles=size_handles, labels=size_labels, 
                              title="RMSD Values", loc="upper left", fontsize=12,
                              title_fontsize=12)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def main():
    # Create output directory
    os.makedirs("modelangelo_plots_outputs", exist_ok=True)
    
    # Model file paths
    model_files = {
        "original": "modelangelo_plots_data/modelangelo_output_original.cif",
        "cryogen": "modelangelo_plots_data/modelangelo_output_cryogen.cif",
        "lowres": "modelangelo_plots_data/modelangelo_output_lowres.cif"
    }
    
    # Verify files exist
    print("\nChecking model files...")
    for name, path in model_files.items():
        exists = os.path.exists(path)
        if not exists:
            print(f"ERROR: File does not exist: {path}")
            return
    
    # Target chains for proximity analysis
    target_chains = ["Aa", "AT", "Ac", "Af"]
    
    print("\nRunning analysis...")
    
    # Extract B-factors for confidence violin plot
    b_factors_all = {name: extract_b_factors(path) for name, path in model_files.items()}
    
    # Generate confidence violin plot
    generate_confidence_violin_plot(b_factors_all, "modelangelo_plots_outputs/confidence_violin_plot.pdf")
    
    # Perform sequence analysis across models
    model_sequences = {}
    for name, path in model_files.items():
        model_sequences[name] = get_chain_sequences(path)
    
    # Find similar chains
    similar_chains = find_similar_chains_across_models(
        model_sequences, model_files, 
        min_length=MIN_CHAIN_LENGTH, min_identity=0.4
    )
    
    # Save fragment alignment results
    print_and_save_results(similar_chains, "modelangelo_plots_outputs/chain_alignments.txt")
    
    # Generate alignment vs identity scatter plot
    generate_alignment_vs_identity_scatter(similar_chains, "modelangelo_plots_outputs/alignment_vs_identity_scatter.pdf")
    
    # Get Original-Cryogen pairs for the target chains
    orig_cryogen_pairs = {}
    
    # First get potential matches from the alignment table
    potential_matches = {}
    for match in similar_chains:
        if match["model1"] == "original" and match["model2"] == "cryogen":
            if match["chain1"] in target_chains:
                if match["chain1"] not in potential_matches:
                    potential_matches[match["chain1"]] = []
                potential_matches[match["chain1"]].append({
                    "cryogen_chain": match["chain2"],
                    "centroid_distance": match["spatial_distance"]["centroid_distance"] if match["spatial_distance"] else float('inf')
                })
        elif match["model1"] == "cryogen" and match["model2"] == "original":
            if match["chain2"] in target_chains:
                if match["chain2"] not in potential_matches:
                    potential_matches[match["chain2"]] = []
                potential_matches[match["chain2"]].append({
                    "cryogen_chain": match["chain1"],
                    "centroid_distance": match["spatial_distance"]["centroid_distance"] if match["spatial_distance"] else float('inf')
                })
    
    # For each original chain, choose the cryogen chain with smallest centroid distance within 3Å
    for orig_chain, matches in potential_matches.items():
        # Sort by centroid distance
        matches.sort(key=lambda x: x["centroid_distance"])
        # Take the closest one that's within 3Å
        for match in matches:
            if match["centroid_distance"] <= 3.0:
                orig_cryogen_pairs[orig_chain] = match["cryogen_chain"]
                break
    
    # Find spatially closest chains from lowres model to original target chains
    closest_to_original = find_closest_chains_by_spatial_proximity(
        model_files["original"], 
        target_chains, 
        model_files["lowres"],
        max_centroid_distance=20.0
    )
    
    # Save closest chains to original results
    save_proximity_results(
        closest_to_original, 
        "original", 
        "lowres", 
        "modelangelo_plots_outputs/lowres_matches_to_original.txt"
    )
    
    print("\nAnalysis complete. All results saved to modelangelo_plots_outputs/ directory.")

if __name__ == "__main__":
    main() 