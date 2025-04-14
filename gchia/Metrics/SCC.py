import os
import numpy as np
import hicstraw
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse
import cooler  # Add cooler import

def init_parser():
    """Initialize argument parser with all necessary parameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix_directory', type=str, help='directory containing the predicted matrix')
    parser.add_argument('--real_hic_filepath', type=str, help='filepath of the real Hi-C matrix')
    parser.add_argument('--chromosome', type=str, help='chromosome to calculate the SCC')
    parser.add_argument('--output_path', type=str, help='path to save the plot')
    parser.add_argument('--max_distance', type=int, default=2000000, help='maximum distance to extract (in base pairs)')
    parser.add_argument('--resolution', type=int, default=10000, help='resolution of the matrix (in base pairs)')
    parser.add_argument('--norm', type=str, default='NONE', help='normalization method')
    parser.add_argument('--log1p', type=str, default='True', help='log1p transform the Hi-C matrix (True/False)')
    return parser

def get_sorted_npy_files(files):
    """
    Sort files by their start coordinate and extract start coordinates
    Parameters:
        files: list of filenames
    Returns:
        tuple: (sorted list of .npy files, list of corresponding start coordinates)
    """
    def extract_start_coordinate(filename):
        # Example filename: pred_matrix_chr10_500000_3000000.npy
        filename = os.path.splitext(filename)[0]  # Remove extension
        parts = filename.split('_')
        # print(f'Parts: {parts}')
        # Find the start coordinate (usually the second-to-last numeric part)
        numeric_parts = [part for part in parts if part.isdigit()]
       # print(f'Numeric parts: {numeric_parts}')
        if len(numeric_parts) >= 2:
            return int(numeric_parts[-2])  # Assuming format is chr_start_end.npy
        
        return float('inf')
    
    # Create a list of (file, start_coordinate) tuples
    file_coords = [(f, extract_start_coordinate(f)) for f in files]
    # Sort by start coordinate
    sorted_file_coords = sorted(file_coords, key=lambda x: x[1])
    # Unzip the sorted list
    sorted_files, start_coords = zip(*sorted_file_coords) if file_coords else ([], [])
    
    return sorted_files, start_coords

def load_matrix(directory, chromosome):
    """
    Load matrix from .npy files in a directory
    Parameters:
        directory: directory containing .npy files
        chromosome: chromosome identifier
    Returns:
        tuple: (list of loaded matrices, list of start coordinates in bin units)
    """
    chromosome_ = chromosome + '_'
    files = [f for f in os.listdir(directory) if f.endswith('.npy') and chromosome_ in f]
    sorted_files, start_coords = get_sorted_npy_files(files)
    sorted_filepaths = [os.path.join(directory, f) for f in sorted_files]
    matrix = [np.load(f) for f in sorted_filepaths]
    # print(f'Loaded {len(matrix)} matrices with start coordinates: {start_coords}')
    return matrix, start_coords

def load_hic_matrix(filepath, chromosome, resolution=10000, norm='NONE', maxdistance=2000000, log1p=True):
    """
    Load Hi-C matrix from various file formats (.hic, .cool, .mcool)
    
    Parameters:
        filepath: path to Hi-C file (.hic, .cool, or .mcool)
        chromosome: chromosome identifier
        resolution: matrix resolution in base pairs
        norm: normalization method
        maxdistance: maximum distance to consider
        log1p: whether to apply log1p transformation
        
    Returns:
        sparse matrix in CSR format
    """
    # Determine file format based on extension
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext in ['.cool', '.mcool']:
        # Use cooler for .cool and .mcool files
        try:
            # For mcool files, specify resolution
            if file_ext == '.mcool':
                clr = cooler.Cooler(f'{filepath}::/resolutions/{resolution}')
            else:
                clr = cooler.Cooler(filepath)
                
            # Handle chromosome name format (some files use 'chr1', others use '1')
            chr_name = chromosome
            if chr_name not in clr.chromnames:
                if chr_name.startswith('chr'):
                    alt_chr_name = chr_name[3:]  # Remove 'chr' prefix
                else:
                    alt_chr_name = f'chr{chr_name}'  # Add 'chr' prefix
                
                if alt_chr_name in clr.chromnames:
                    chr_name = alt_chr_name
                else:
                    print(f"Warning: {chromosome} not found in cooler file. Available chromosomes: {clr.chromnames}")
                    raise ValueError(f"Chromosome {chromosome} not found in cooler file")
            
            # Fetch the full chromosome matrix
            matrix = clr.matrix(balance=norm != 'NONE').fetch(chr_name)
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            # Convert to sparse format
            binX, binY = np.nonzero(matrix)
            counts = matrix[binX, binY]
            
            # Filter by distance if maxdistance is specified
            if maxdistance is not None:
                max_bin_distance = maxdistance // resolution
                distance_mask = np.abs(binX - binY) <= max_bin_distance
                binX = binX[distance_mask]
                binY = binY[distance_mask]
                counts = counts[distance_mask]
            
            # Apply log1p transformation if requested
            if log1p:
                counts = np.log1p(counts)
                
            hic_matrix = coo_matrix((counts, (binX, binY)), shape=matrix.shape)
            return hic_matrix.tocsr()
            
        except Exception as e:
            print(f'Error reading cooler matrix for {chromosome}: {str(e)}')
            raise
    else:
        # Default to hicstraw for .hic files
        chromosome = chromosome.replace('chr', '')
        try:
            result = hicstraw.straw('observed', norm, filepath, f'{chromosome}', f'{chromosome}', 'BP', resolution)
        except Exception as e:
            print(f'Error reading Hi-C matrix for {chromosome}: {str(e)}')
            raise
        
        binX = []
        binY = []
        counts = []
        
        for entry in result:
            if maxdistance is None or abs(entry.binX - entry.binY) <= maxdistance:
                binX.append(entry.binX // resolution)
                binY.append(entry.binY // resolution)
                counts.append(entry.counts)
                
        binX = np.array(binX)
        binY = np.array(binY)
        counts = np.array(counts)
        if log1p:
            counts = np.log1p(counts)
        hic_matrix = coo_matrix((counts, (binX, binY)), shape=(binX.max() + 1, binY.max() + 1))
        return hic_matrix.tocsr()

def merge_and_extract_diagonals(matrix_list, start_coords, max_distance=2000000, resolution=10000):
    """
    Extract and merge diagonals based on actual start coordinates
    Parameters:
        matrix_list: list of input matrices
        start_coords: list of start coordinates in base pairs
        max_distance: maximum distance in base pairs
        resolution: resolution in base pairs
    Returns:
        tuple: (merged_diagonals, counts_diagonals)
    """
    max_distance_bin = max_distance // resolution
    
    # Convert start coordinates to bin units
    start_bins = [coord // resolution for coord in start_coords]
    
    # Find the total size needed for the merged matrix
    matrix_sizes = [matrix.shape[0] for matrix in matrix_list]
    end_bins = [start + size for start, size in zip(start_bins, matrix_sizes)]
    # print(f'End bins: {end_bins}')
    total_size = max(end_bins) if end_bins else 0
    print(f'Total size: {total_size}')
    # Pre-allocate arrays for results and counts
    merged_diagonals = [np.zeros(total_size - d) for d in range(max_distance_bin + 1)]
    counts_diagonals = [np.zeros(total_size - d) for d in range(max_distance_bin + 1)]
    
    # Process each matrix
    for i, matrix in enumerate(matrix_list):
        start_bin = start_bins[i]
        
        # Convert to CSR format for efficient diagonal extraction
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
            
        # Extract diagonals
        for d in range(max_distance_bin + 1):
            diagonal = matrix.diagonal(k=d)
            
            # Process each value in the diagonal
            for j, val in enumerate(diagonal):
                pos = start_bin + j
                if pos < len(merged_diagonals[d]):
                    merged_diagonals[d][pos] += val
                    counts_diagonals[d][pos] += 1
    
    # Calculate averages for positions with counts > 0
    for d in range(max_distance_bin + 1):
        mask = counts_diagonals[d] > 0
        merged_diagonals[d][mask] /= counts_diagonals[d][mask]
    
    return merged_diagonals, counts_diagonals

def calculate_pearson_for_diagonals(predicted_diagonals, counts_diagonals, real_diagonals):
    """
    Calculate Pearson correlation for each diagonal, only considering regions with non-zero counts
    Parameters:
        predicted_diagonals: list of predicted diagonal arrays
        counts_diagonals: list of count arrays indicating predicted positions
        real_diagonals: list of real diagonal arrays
    Returns:
        list of correlation coefficients
    """
    pearson_coeffs = []
    
    for pred_diag, counts_diag, real_diag in zip(predicted_diagonals, counts_diagonals, real_diagonals):
        
        min_length = min(len(pred_diag), len(real_diag), len(counts_diag))
        pred_diag = pred_diag[:min_length]
        real_diag = real_diag[:min_length]
        counts_diag = counts_diag[:min_length]
        
       
        mask = counts_diag > 0
        
       
        if not np.any(mask):
            pearson_coeffs.append(np.nan)
            continue
            
        
        pred_valid = pred_diag[mask]
        real_valid = real_diag[mask]
        
        if len(pred_valid) > 1:
           
            coeff, _ = pearsonr(pred_valid, real_valid)
        else:
            coeff = np.nan
        
        pearson_coeffs.append(coeff)
    
    return pearson_coeffs

def main():
    """Main function to calculate SCC"""
    parser = init_parser()
    args = parser.parse_args()
    
    # Load predicted matrices
    print('Loading matrices')
    matrix, start_coords = load_matrix(args.matrix_directory, args.chromosome)
    
    # Extract and merge diagonals
    print('Extracting and merging diagonals')
    predicted_diagonals, counts_diagonals = merge_and_extract_diagonals(
        matrix,
        start_coords,
        max_distance=args.max_distance,
        resolution=args.resolution
    )
    
    log1p_bool = args.log1p.lower() == 'true'
    
    
    real_matrix = load_hic_matrix(
        args.real_hic_filepath,
        args.chromosome,
        resolution=args.resolution,
        norm=args.norm,
        maxdistance=args.max_distance,
        log1p=log1p_bool
    )
    
    # Extract diagonals from real matrix
    print('Extracting real Hi-C matrix diagonals')
    real_diagonals = []
    for d in range(args.max_distance // args.resolution + 1):
        real_diagonals.append(real_matrix.diagonal(k=d))
    
    # Calculate correlations
    print('Calculating Pearson correlation coefficients')
    pearson_coeffs = calculate_pearson_for_diagonals(predicted_diagonals, counts_diagonals, real_diagonals)
    
    # Save results and plot
    if args.output_path:
        np.save(args.output_path, pearson_coeffs)
        print('Plotting correlation coefficients')
        plot_pearson_coefficients(
            pearson_coeffs,
            resolution=args.resolution,
            output_path=args.output_path
        )
    
    return pearson_coeffs

def plot_pearson_coefficients(pearson_coeffs, resolution=10000, output_path=None):
    """
    Plot Pearson correlation coefficients
    Parameters:
        pearson_coeffs: list of correlation coefficients
        resolution: matrix resolution in base pairs
        output_path: path to save the plot
    """
    distances = np.arange(len(pearson_coeffs)) * resolution / 1000
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances, pearson_coeffs, '-', linewidth=2)
    plt.xlabel('Genomic Distance (kb)')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('Correlation vs Genomic Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    main()