# merge the predicted matrix into a big matrix
# and save the big matrix into a file
import os
import numpy as np
import subprocess
import sys
import argparse
import cooler
def get_sorted_npy_files(files):
    """
    Sort files by their start coordinate
    Parameters:
        files: list of filenames
    Returns:
        sorted list of .npy files
    """
    def extract_start_coordinate(filename):
        # Example filename: pred_matrix_chr10_500000_3000000.npy
        parts = filename.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return float('inf')
    return sorted(files, key=extract_start_coordinate)



def load_matrix(directory, chromosome):
    """
    Load matrix from .npy files in a directory
    Parameters:
        directory: directory containing .npy files
        chromosome: chromosome identifier
    Returns:
        tuple: (list of loaded matrices, list of start positions)
    """
    chromosome_ = chromosome + '_'
    files = [f for f in os.listdir(directory) if f.endswith('.npy') and chromosome_ in f]
    sorted_files = get_sorted_npy_files(files)
    sorted_filepaths = [os.path.join(directory, f) for f in sorted_files]
    
    matrices = []
    start_positions = []
    
    for filepath in sorted_filepaths:
        # Extract start position from filename
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        start_pos = None
        for part in parts:
            if part.isdigit():
                start_pos = int(part)
                break
        
        if start_pos is not None:
            matrix = np.load(filepath)
            matrix = np.expm1(matrix)  # Convert back to count matrix
            matrices.append(matrix)
            start_positions.append(start_pos)
    
    print(f'Loaded {len(matrices)} matrices')
    return matrices, start_positions

def convert_to_coo_format(matrices, start_positions, chrom, resolution, output_directory,
                               threshold=0.05, overlap_method='mean'):
    """
    Convert matrices to cooler load format and write to file
    Parameters:
        matrices: list of predicted matrices
        start_positions: list of start positions for each matrix
        chromosome: chromosome identifier
        resolution: resolution in base pairs
        output_directory: directory to save the output files
        threshold: threshold to filter out small values
        overlap_method: method to handle overlapping regions ('mean' or 'max')
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    output_file = os.path.join(output_directory, f"{chrom}.txt")
    
    # Convert start positions to bin indices
    start_bins = [pos // resolution for pos in start_positions]
    
    # Track processed positions
    processed_contacts = {}
    contact_counts = {}  # For counting when using 'mean' method
    
    for matrix_idx, (matrix, start_pos) in enumerate(zip(matrices, start_positions)):
        matrix_size = matrix.shape[0]
        start_bin = start_bins[matrix_idx]
        
        # Apply threshold to filter out small values
        filtered_matrix = np.where(matrix > threshold, matrix, 0)
        # only upper triangle is needed
        filtered_matrix = np.triu(filtered_matrix)
        # Process non-zero elements
        non_zero_indices = np.nonzero(filtered_matrix)
        for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
            # Calculate global bin positions (as bin indices)
            global_i = start_bin + i
            global_j = start_bin + j
            contact_key = (min(global_i, global_j), max(global_i, global_j))
            
            # Current contact value
            value = filtered_matrix[i, j]
            
            # Handle overlapping regions based on chosen method
            if contact_key in processed_contacts:
                if overlap_method == 'max':
                    processed_contacts[contact_key] = max(processed_contacts[contact_key], value)
                elif overlap_method == 'mean':
                    processed_contacts[contact_key] += value
                    contact_counts[contact_key] += 1
            else:
                processed_contacts[contact_key] = value
                if overlap_method == 'mean':
                    contact_counts[contact_key] = 1
        
        print(f"Processed matrix {matrix_idx+1}/{len(matrices)}")
    
    # Calculate mean values if using mean method
    if overlap_method == 'mean':
        for key in processed_contacts:
            if contact_counts[key] > 1:
                processed_contacts[key] /= contact_counts[key]
    
    # Write all unique contacts to file
    contact_count = 0
    with open(output_file, 'w') as f:
        for (bin_i, bin_j), value in processed_contacts.items():
            if value > 0:  # Ensure value is positive
                # hicpeaks format requires bin indices, not coordinates
                f.write(f"{bin_i}\t{bin_j}\t{value}\n")
                contact_count += 1
    
    print(f"Total unique contacts written: {contact_count} to {output_file}")


def convert_to_bg2_format(matrices, start_positions, chrom, resolution, output_directory,
                          threshold=0.05, overlap_method='mean'):
    """
    Convert matrices to BG2 format and write to file
    Parameters:
        matrices: list of predicted matrices
        start_positions: list of start positions for each matrix
        chrom: chromosome identifier
        resolution: resolution in base pairs
        output_directory: directory to save the output files
        threshold: threshold to filter out small values
        overlap_method: method to handle overlapping regions ('mean' or 'max')
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Use .bg2 extension to clearly indicate format
    output_file = os.path.join(output_directory, f"{chrom}.bg2")
    
    # Convert start positions to bin indices
    start_bins = [pos // resolution for pos in start_positions]
    
    # Track processed positions
    processed_contacts = {}
    contact_counts = {}  # For counting when using 'mean' method
    
    for matrix_idx, (matrix, start_pos) in enumerate(zip(matrices, start_positions)):
        matrix_size = matrix.shape[0]
        start_bin = start_bins[matrix_idx]
        
        # Apply threshold to filter out small values
        filtered_matrix = np.where(matrix > threshold, matrix, 0)
        # only upper triangle is needed
        filtered_matrix = np.triu(filtered_matrix)
        # Process non-zero elements
        non_zero_indices = np.nonzero(filtered_matrix)
        for i, j in zip(non_zero_indices[0], non_zero_indices[1]):
            # Calculate global bin positions (as bin indices)
            global_i = start_bin + i
            global_j = start_bin + j
            contact_key = (min(global_i, global_j), max(global_i, global_j))
            
            # Current contact value
            value = filtered_matrix[i, j]
            
            # Handle overlapping regions based on chosen method
            if contact_key in processed_contacts:
                if overlap_method == 'max':
                    processed_contacts[contact_key] = max(processed_contacts[contact_key], value)
                elif overlap_method == 'mean':
                    processed_contacts[contact_key] += value
                    contact_counts[contact_key] += 1
            else:
                processed_contacts[contact_key] = value
                if overlap_method == 'mean':
                    contact_counts[contact_key] = 1
        
        print(f"Processed matrix {matrix_idx+1}/{len(matrices)}")
    
    # Calculate mean values if using mean method
    if overlap_method == 'mean':
        for key in processed_contacts:
            if contact_counts[key] > 1:
                processed_contacts[key] /= contact_counts[key]
    
    # Write all unique contacts to file in BG2 format
    contact_count = 0
    with open(output_file, 'w') as f:
        for (bin_i, bin_j), value in processed_contacts.items():
            # value_int = round(value)
            if value > 0:  # Ensure value is positive
                
                # Convert bin indices to genomic coordinates
                start1 = bin_i * resolution
                end1 = start1 + resolution
                start2 = bin_j * resolution
                end2 = start2 + resolution
                
                # BG2 format: chrom1 start1 end1 chrom2 start2 end2 count
                f.write(f"{chrom}\t{start1}\t{end1}\t{chrom}\t{start2}\t{end2}\t{value}\n")
                contact_count += 1
    
    print(f"Total unique contacts written: {contact_count} to {output_file}")
    return output_file

def create_cool_file_from_bg2(bg2_file, chrom_sizes, resolution, output_cool):
    """
    Create a .cool file from a BG2 format file
    
    Parameters:
        bg2_file: Path to the BG2 format file
        chrom_sizes: Path to the chromosome sizes file
        resolution: Resolution in base pairs
        output_cool: Path for the output .cool file
    """
    cmd = [
        "cooler", "load", 
        "-f", "bg2",
        "--count-as-float",  # 直接添加此参数以支持浮点数计数值
        f"{chrom_sizes}:{resolution}",
        bg2_file,
        output_cool
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Created cool file: {output_cool}")
    return output_cool

def process_chromosome(input_dir, chrom, resolution, output_dir, chrom_sizes, threshold=0.05, overlap_method='mean'):
    """
    Process a single chromosome and create a cool file
    
    Parameters:
        input_dir: Directory containing prediction matrices
        chrom: Chromosome name (e.g., 'chr10')
        resolution: Resolution in base pairs
        output_dir: Output directory
        chrom_sizes: Path to chromosome sizes file
        threshold: Threshold to filter out small values
        overlap_method: Method to handle overlapping regions
    
    Returns:
        Path to created cool file
    """
    # Load matrices for the chromosome
    matrices, start_positions = load_matrix(input_dir, chrom)
    
    if not matrices or not start_positions:
        print(f"No matrix files found for chromosome {chrom} in {input_dir}")
        return None
    
    # Convert to BG2 format
    bg2_file = convert_to_bg2_format(
        matrices, start_positions, chrom, resolution, output_dir, 
        threshold=threshold, overlap_method=overlap_method)
    
    # Create cool file
    cool_output = os.path.join(output_dir, f"{chrom}.cool")
    created_cool = create_cool_file_from_bg2(bg2_file, chrom_sizes, resolution, cool_output)

    # addweight
    add_weight_column(created_cool)
    
    return created_cool

def add_weight_column(fcool):
    '''
    all bins' bias set to 1.0
    '''
    clr = cooler.Cooler(fcool)
    n_bins = clr.bins().shape[0]

    if 'weight' not in clr.bins().columns:
        h5opts = dict(compression='gzip', compression_opts=6)
        with clr.open('r+') as f:
            # Create a weight column
            f['bins'].create_dataset('weight', data=np.ones(n_bins), **h5opts)



def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Convert predicted matrices to cool format')
    
    parser.add_argument('input_dir', 
                        help='Directory containing predicted matrices (.npy files)')
    
    parser.add_argument('--output_dir', '-o', 
                        help='Output directory (default: input_dir/../cool_output)')
    
    parser.add_argument('--chrom_sizes', '-c', required=True,
                        help='Path to chromosome sizes file')
    
    parser.add_argument('--chromosomes', '-chr', nargs='+', 
                        help='Chromosomes to process (e.g., chr1 chr2 chr3). If not provided, all chromosomes in input directory are processed.')
    
    parser.add_argument('--resolution', '-r', type=int, default=5000,
                        help='Resolution in base pairs (default: 5000)')
    
    parser.add_argument('--threshold', '-t', type=float, default=0.05,
                        help='Threshold to filter out small values (default: 0.05)')
    
    parser.add_argument('--overlap_method', '-m', choices=['mean', 'max'], default='mean',
                        help='Method to handle overlapping regions (default: mean)')
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    args = parse_arguments()
    
    # Input directory must exist
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Chromosome sizes file must exist
    if not os.path.isfile(args.chrom_sizes):
        print(f"Error: Chromosome sizes file {args.chrom_sizes} does not exist")
        sys.exit(1)
    
    # Set default output directory if not provided
    if args.output_dir is None:
        parent_dir = os.path.dirname(os.path.abspath(args.input_dir))
        args.output_dir = os.path.join(parent_dir, "cool_output")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Determine chromosomes to process
    if args.chromosomes:
        chromosomes = args.chromosomes
    else:
        # Extract unique chromosome names from .npy files in input directory
        all_files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
        chromosomes = set()
        for f in all_files:
            parts = f.split('_')
            for i, part in enumerate(parts):
                if part.startswith('chr'):
                    chromosomes.add(part)
                    break
        chromosomes = sorted(list(chromosomes))
    
    if not chromosomes:
        print("Error: No chromosomes found to process")
        sys.exit(1)
    
    print(f"Chromosomes to process: {chromosomes}")
    
    # Process each chromosome
    cool_files = []
    for chrom in chromosomes:
        print(f"\nProcessing chromosome {chrom}...")
        cool_file = process_chromosome(
            args.input_dir, chrom, args.resolution, args.output_dir, args.chrom_sizes,
            threshold=args.threshold, overlap_method=args.overlap_method)
        if cool_file:
            cool_files.append(cool_file)
    
    if not cool_files:
        print("No cool files were created")
        sys.exit(1)
    
    print("\nAll cool files created successfully:")
    for f in cool_files:
        print(f"  {f}")
    
    # Optionally merge cool files if multiple chromosomes were processed
    if len(cool_files) > 1:
        print("\nMultiple chromosomes were processed. You can merge them using:")
        merged_cool = os.path.join(args.output_dir, "merged.cool")
        merge_cmd = f"cooler merge {merged_cool} {' '.join(cool_files)}"
        print(f"  {merge_cmd}")

if __name__ == "__main__":
    main()

